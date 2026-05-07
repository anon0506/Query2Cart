"""Generic task generation pipeline driven by DomainConfig.

Replaces the hardcoded laptop-specific ``02_generate_tasks.py`` with a
domain-agnostic implementation.  Every constraint name, sampling value,
coherence rule, and difficulty bracket comes from the config — nothing is
hardcoded to a particular product domain.

Architecture (same as the laptop pipeline):
  1. Sample constraint set from config's constraint registry (no LLM)
  2. Apply constraints → filtered pool; validate pool size
  3. Pick anchor product from pool (weighted by completeness + popularity)
  4. LLM call 1: generates initial_query + preferences + behavioral_profile
  5. Validate query + preferences
  6. LLM call 2: generates revelation_plan (with few-shot examples for
     difficulty compliance); retries up to 3× on difficulty mismatch
     without re-generating the profile
  7. Assign difficulty based on revelation plan structure
  8. Build final task object with utility scoring
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from shared.config import ConstraintSpec, DifficultyBracket, DomainConfig, ConstraintOp
from shared.filter import GenericFilter, OPERATORS
from shared.llm import call_llm_json, format_prompt_template, resolve_prompt_path
from shared.scoring import (
    compute_preference_weights,
    compute_pool_stats,
    compute_product_utility,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diversity tracking
# ---------------------------------------------------------------------------

class DiversityTracker:
    """Track constraint, value, and preference usage across accepted tasks.

    When sampling, boosts underrepresented constraints and decays weights
    for values that have already been used, so the final task set covers
    as many constraint dimensions and values as possible.

    Preference diversity tracking records which (attribute, direction) pairs
    have been used and provides hints to the LLM to pick underrepresented
    combinations.
    """

    def __init__(self, config: DomainConfig):
        self.constraint_counts: dict[str, int] = {}
        self.value_counts: dict[str, dict[str, int]] = {}
        for c in config.constraints:
            if c.desc_constraint or not c.sampling_values:
                continue
            self.constraint_counts[c.name] = 0
            self.value_counts[c.name] = {str(v): 0 for v in c.sampling_values}
        self.total_tasks = 0

        self.pref_combo_counts: dict[str, int] = {}
        self.pref_pair_counts: dict[tuple[str, str], int] = {}
        self._all_pref_pairs: list[tuple[str, str]] = []
        for pa in config.preference_attributes:
            for d in pa.directions:
                self._all_pref_pairs.append((pa.attribute, d))

    def record(self, constraints: dict[str, Any], preferences: list[dict] | None = None) -> None:
        """Record an accepted task's constraint set and preferences."""
        self.total_tasks += 1
        for name, value in constraints.items():
            if name in self.constraint_counts:
                self.constraint_counts[name] += 1
            if name in self.value_counts:
                key = str(value)
                if key in self.value_counts[name]:
                    self.value_counts[name][key] += 1

        if preferences:
            combo_key = self._pref_combo_key(preferences)
            self.pref_combo_counts[combo_key] = self.pref_combo_counts.get(combo_key, 0) + 1
            for pref in preferences:
                attr = pref.get("attribute", "")
                direction = pref.get("direction", "")
                pair = (attr, direction)
                self.pref_pair_counts[pair] = self.pref_pair_counts.get(pair, 0) + 1

    @staticmethod
    def _pref_combo_key(preferences: list[dict]) -> str:
        """Canonical string key for a preference combination."""
        pairs = sorted(
            (p.get("attribute", ""), p.get("direction", ""))
            for p in preferences
        )
        return "|".join(f"{a}:{d}" for a, d in pairs)

    def sample_preference_pairs(
        self,
        config: DomainConfig,
    ) -> list[tuple[str, str]]:
        """Sample (attribute, direction) pairs with diversity-aware weighting.

        Picks 2 or 3 pairs (no duplicate attributes), preferring underused
        combinations.  Called before the LLM so that preference selection is
        deterministic and diverse — the LLM only fills in natural-language
        details (priority, trigger, description).
        """
        all_pairs: list[tuple[str, str]] = list(self._all_pref_pairs)
        if len(all_pairs) < 2:
            return list(all_pairs)

        unique_attrs = {a for a, _ in all_pairs}
        n_prefs = random.choice([2, 3]) if len(unique_attrs) >= 3 else 2

        selected: list[tuple[str, str]] = []
        remaining = list(all_pairs)

        for _ in range(n_prefs):
            if not remaining:
                break
            weights = [
                1.0 / (1.0 + self.pref_pair_counts.get(p, 0))
                for p in remaining
            ]
            pick = random.choices(range(len(remaining)), weights=weights, k=1)[0]
            chosen = remaining[pick]
            selected.append(chosen)
            chosen_attr = chosen[0]
            remaining = [p for p in remaining if p[0] != chosen_attr]

        return selected

    def get_value_weights(
        self,
        constraint_name: str,
        values: list[Any],
        base_weights: list[float],
    ) -> list[float]:
        """Adjust value weights: prefer less-used values."""
        if self.total_tasks == 0:
            return list(base_weights)
        usage = self.value_counts.get(constraint_name, {})
        return [
            w / (1.0 + usage.get(str(v), 0) * 0.5)
            for v, w in zip(values, base_weights)
        ]

    def get_constraint_boost(self, constraint_name: str, base_prob: float) -> float:
        """Boost underrepresented constraints, dampen overrepresented ones."""
        if self.total_tasks < 5:
            return base_prob
        usage_rate = self.constraint_counts.get(constraint_name, 0) / self.total_tasks
        if usage_rate < base_prob * 0.5:
            return min(base_prob * 1.5, 0.95)
        if usage_rate > base_prob * 2.0:
            return max(base_prob * 0.7, 0.05)
        return base_prob


# ---------------------------------------------------------------------------
# Pool index (bitmask pre-computation)
# ---------------------------------------------------------------------------

class PoolIndex:
    """Bitmask index over a catalog for O(1) pool-size queries.

    For every (constraint_name, value) pair, stores a numpy boolean array
    of length N indicating which catalog rows match.  Combining constraints
    is bitwise AND; ``popcount`` (np.count_nonzero) gives pool size instantly.
    """

    def __init__(self, catalog: pd.DataFrame, config: DomainConfig):
        self._n = len(catalog)
        self._full_mask = np.ones(self._n, dtype=bool)
        self._index: dict[tuple[str, Any], np.ndarray] = {}
        self._constraint_any_mask: dict[str, np.ndarray] = {}

        for cspec in config.constraints:
            if cspec.desc_constraint or not cspec.sampling_values:
                continue
            if cspec.attribute not in catalog.columns:
                continue

            op_fn = OPERATORS.get(cspec.operator)
            if op_fn is None:
                continue

            col = catalog[cspec.attribute]
            any_viable = np.zeros(self._n, dtype=bool)

            for val in cspec.sampling_values:
                mask = op_fn(col, val).values.astype(bool)
                self._index[(cspec.name, val)] = mask
                any_viable |= mask

            self._constraint_any_mask[cspec.name] = any_viable

    @property
    def catalog_size(self) -> int:
        return self._n

    def full_mask(self) -> np.ndarray:
        return self._full_mask.copy()

    def get_mask(self, constraint_name: str, value: Any) -> np.ndarray:
        return self._index.get((constraint_name, value), np.zeros(self._n, dtype=bool))

    def pool_size(self, mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    def pool_ids(self, mask: np.ndarray, catalog: pd.DataFrame, id_column: str) -> list:
        return catalog.loc[mask, id_column].tolist()

    def constraint_viable(self, constraint_name: str, current_mask: np.ndarray) -> bool:
        """True if at least one sampling value for this constraint produces a non-empty pool."""
        any_mask = self._constraint_any_mask.get(constraint_name)
        if any_mask is None:
            return False
        return np.any(current_mask & any_mask)

    def has_value(self, constraint_name: str, value: Any) -> bool:
        return (constraint_name, value) in self._index


# ---------------------------------------------------------------------------
# Constraint sampling (config-driven, diversity-aware, pool-index-guided)
# ---------------------------------------------------------------------------

def _sample_one_constraint_set(
    config: DomainConfig,
    target_bracket: DifficultyBracket | None = None,
    diversity: DiversityTracker | None = None,
    value_frequencies: dict[str, dict[Any, float]] | None = None,
) -> dict[str, Any]:
    """Sample a constraint set using count-first selection + frequency-weighted values.

    1. Decide how many constraints to include (from bracket or default 3-6).
    2. Pre-select always-include constraints.
    3. Fill remaining slots by weighted sampling without replacement.
    4. For each selected constraint, pick a value weighted by catalog frequency
       and diversity (prefer less-used values).
    """
    if target_bracket is not None:
        min_c = target_bracket.min_constraints
        max_c = min_c + 2
    else:
        min_c, max_c = 3, 6
    n_target = random.randint(min_c, max_c)

    always: list[ConstraintSpec] = []
    candidates: list[ConstraintSpec] = []
    for c in config.constraints:
        if c.desc_constraint or not c.sampling_values:
            continue
        if c.always_include:
            always.append(c)
        else:
            candidates.append(c)

    selected = list(always)
    remaining = n_target - len(selected)

    if remaining > 0 and candidates:
        weights = []
        for c in candidates:
            w = c.sampling_probability
            if diversity:
                w = diversity.get_constraint_boost(c.name, w)
            weights.append(max(w, 0.01))

        available = list(range(len(candidates)))
        for _ in range(min(remaining, len(available))):
            if not available:
                break
            cur_w = [weights[i] for i in available]
            total = sum(cur_w)
            if total <= 0:
                break
            probs = [w / total for w in cur_w]
            pick = random.choices(range(len(available)), weights=probs, k=1)[0]
            selected.append(candidates[available.pop(pick)])

    constraints: dict[str, Any] = {}
    for cspec in selected:
        vals = list(cspec.sampling_values)
        if not vals:
            continue

        if value_frequencies and cspec.name in value_frequencies:
            freq = value_frequencies[cspec.name]
            base_w = [max(freq.get(v, 0.001), 0.001) for v in vals]
        else:
            base_w = [1.0] * len(vals)

        if diversity:
            base_w = diversity.get_value_weights(cspec.name, vals, base_w)

        constraints[cspec.name] = random.choices(vals, weights=base_w, k=1)[0]

    return constraints


def sample_constraint_set(
    catalog: pd.DataFrame,
    config: DomainConfig,
    difficulty: str,
    max_retries: int = 20,
    diversity: DiversityTracker | None = None,
    value_frequencies: dict[str, dict[Any, float]] | None = None,
    pool_index: PoolIndex | None = None,
) -> Optional[tuple[dict[str, Any], pd.DataFrame]]:
    """Sample a coherent constraint set whose pool lands in the target bracket.

    When a ``pool_index`` is provided, uses guided single-pass construction:
    constraints are selected, then values are picked incrementally with pool
    awareness.  Falls back to blind retry when no index is available.
    """
    bracket = config.difficulty.get(difficulty)
    if bracket is None:
        return None
    lo, hi = bracket.pool_range
    min_constraints = bracket.min_constraints

    if pool_index is not None:
        return _guided_constraint_sampling(
            catalog, config, bracket, difficulty, pool_index,
            diversity=diversity, value_frequencies=value_frequencies,
            max_retries=max_retries,
        )

    # Fallback: blind retry (legacy path)
    for _ in range(max_retries * 4):
        constraints = _sample_one_constraint_set(
            config, bracket, diversity, value_frequencies,
        )
        if len(constraints) < min_constraints:
            continue
        if not config.is_coherent(constraints):
            continue
        pool = GenericFilter.apply(catalog, constraints, config.constraints)
        if lo <= len(pool) <= hi:
            return constraints, pool

    return None


def _guided_constraint_sampling(
    catalog: pd.DataFrame,
    config: DomainConfig,
    bracket: DifficultyBracket,
    difficulty: str,
    pool_index: PoolIndex,
    *,
    diversity: DiversityTracker | None = None,
    value_frequencies: dict[str, dict[Any, float]] | None = None,
    max_retries: int = 20,
) -> Optional[tuple[dict[str, Any], pd.DataFrame]]:
    """Guided single-pass constraint sampling using the pool index.

    1. Select which constraints to include (always_include + weighted sampling).
    2. Order constraints by restrictiveness (ascending for easy/medium, descending for hard).
    3. Pick values incrementally, checking pool size after each.
    """
    lo, hi = bracket.pool_range
    mid = (lo + hi) / 2
    min_c = bracket.min_constraints
    max_c = min_c + 2

    active = [c for c in config.constraints if not c.desc_constraint and c.sampling_values]
    always = [c for c in active if c.always_include]
    candidates = [c for c in active if not c.always_include]

    for _attempt in range(max_retries):
        n_target = random.randint(min_c, max_c)

        # Step 1: select which constraints to include
        selected = list(always)
        remaining_slots = n_target - len(selected)

        if remaining_slots > 0 and candidates:
            current_mask = pool_index.full_mask()
            viable_candidates = [
                c for c in candidates
                if pool_index.constraint_viable(c.name, current_mask)
            ]

            weights = []
            for c in viable_candidates:
                w = c.sampling_probability
                if diversity:
                    w = diversity.get_constraint_boost(c.name, w)
                weights.append(max(w, 0.01))

            available = list(range(len(viable_candidates)))
            for _ in range(min(remaining_slots, len(available))):
                if not available:
                    break
                cur_w = [weights[i] for i in available]
                total = sum(cur_w)
                if total <= 0:
                    break
                probs = [w / total for w in cur_w]
                pick = random.choices(range(len(available)), weights=probs, k=1)[0]
                selected.append(viable_candidates[available.pop(pick)])

        if len(selected) < min_c:
            continue

        # Step 2: order by restrictiveness
        is_hard = difficulty in ("large", "oc_feasible", "oc_infeasible")
        selected.sort(
            key=lambda c: _constraint_restrictiveness(c, pool_index),
            reverse=is_hard,
        )

        # Step 3: pick values incrementally
        current_mask = pool_index.full_mask()
        constraints: dict[str, Any] = {}
        success = True

        for cspec in selected:
            vals = list(cspec.sampling_values)
            if not vals:
                continue

            # Compute candidate pool sizes
            candidate_sizes: list[tuple[Any, int, np.ndarray]] = []
            for v in vals:
                v_mask = current_mask & pool_index.get_mask(cspec.name, v)
                candidate_sizes.append((v, pool_index.pool_size(v_mask), v_mask))

            # Filter to viable values: those that keep pool achievable
            remaining_constraints = len(selected) - len(constraints) - 1
            viable = [
                (v, sz, m) for v, sz, m in candidate_sizes
                if sz > 0 or (lo == 0 and remaining_constraints == 0)
            ]

            if not viable:
                success = False
                break

            # Weight viable values: frequency * diversity_decay * pool_steering
            if value_frequencies and cspec.name in value_frequencies:
                freq = value_frequencies[cspec.name]
                base_w = [max(freq.get(v, 0.001), 0.001) for v, _, _ in viable]
            else:
                base_w = [1.0] * len(viable)

            if diversity:
                base_w = diversity.get_value_weights(
                    cspec.name, [v for v, _, _ in viable], base_w,
                )

            # Pool steering bonus: prefer values that move pool toward bracket midpoint
            for i, (v, sz, _) in enumerate(viable):
                if hi > 0:
                    distance = abs(sz - mid) / max(hi, 1)
                    base_w[i] *= max(0.3, 1.0 - distance * 0.5)

            chosen_idx = random.choices(range(len(viable)), weights=base_w, k=1)[0]
            chosen_val, _, chosen_mask = viable[chosen_idx]

            constraints[cspec.name] = chosen_val
            current_mask = chosen_mask

        if not success:
            continue

        if not config.is_coherent(constraints):
            continue

        final_size = pool_index.pool_size(current_mask)
        if lo <= final_size <= hi:
            pool = catalog.loc[current_mask]
            return constraints, pool

    return None


def _constraint_restrictiveness(cspec: ConstraintSpec, pool_index: PoolIndex) -> float:
    """Median hit-rate across sampling values (higher = less restrictive)."""
    if not cspec.sampling_values:
        return 0.5
    n = pool_index.catalog_size
    if n == 0:
        return 0.5
    rates = []
    full = pool_index.full_mask()
    for v in cspec.sampling_values:
        m = pool_index.get_mask(cspec.name, v)
        rates.append(np.count_nonzero(m) / n)
    return float(np.median(rates)) if rates else 0.5


def sample_oc_constraint_set(
    catalog: pd.DataFrame,
    config: DomainConfig,
    infeasible: bool = False,
    max_retries: int = 80,
    diversity: DiversityTracker | None = None,
    value_frequencies: dict[str, dict[Any, float]] | None = None,
    pool_index: PoolIndex | None = None,
) -> Optional[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    """Generate an over-constrained task: valid base + operator-aware tightening.

    Returns ``(oc_constraints, oc_pool, base_pool)`` or ``None``.
    """
    for _ in range(max_retries):
        base = sample_constraint_set(
            catalog, config, "medium", max_retries=10,
            diversity=diversity, value_frequencies=value_frequencies,
            pool_index=pool_index,
        )
        if base is None:
            continue
        constraints, base_pool = base

        tighten_candidates = _build_tighten_candidates(constraints, config, pool_index)

        random.shuffle(tighten_candidates)
        for key, val in tighten_candidates:
            oc = constraints.copy()
            oc[key] = val
            if pool_index is not None:
                mask = pool_index.full_mask()
                for k, v in oc.items():
                    mask = mask & pool_index.get_mask(k, v)
                oc_size = pool_index.pool_size(mask)
                if infeasible and oc_size == 0:
                    oc_pool = catalog.loc[mask]
                    return oc, oc_pool, base_pool
                elif not infeasible and 1 <= oc_size <= 3:
                    oc_pool = catalog.loc[mask]
                    return oc, oc_pool, base_pool
            else:
                oc_pool = GenericFilter.apply(catalog, oc, config.constraints)
                if infeasible and len(oc_pool) == 0:
                    return oc, oc_pool, base_pool
                elif not infeasible and 1 <= len(oc_pool) <= 3:
                    return oc, oc_pool, base_pool

    return None


def _build_tighten_candidates(
    constraints: dict[str, Any],
    config: DomainConfig,
    pool_index: PoolIndex | None = None,
) -> list[tuple[str, Any]]:
    """Build operator-aware tightening candidates for OC tasks."""
    candidates: list[tuple[str, Any]] = []

    for cspec in config.get_filterable_constraints():
        if not cspec.sampling_values:
            continue

        sorted_vals = sorted(cspec.sampling_values, key=lambda v: (
            float(v) if isinstance(v, (int, float)) else 0
        ))
        op = cspec.operator

        if cspec.name in constraints:
            current = constraints[cspec.name]

            if op == ConstraintOp.LTE:
                tighter = [v for v in sorted_vals if _lt(v, current)]
                if tighter:
                    candidates.append((cspec.name, tighter[0]))

            elif op == ConstraintOp.GTE:
                tighter = [v for v in reversed(sorted_vals) if _gt(v, current)]
                if tighter:
                    candidates.append((cspec.name, tighter[0]))

            elif op == ConstraintOp.RANGE:
                tighter = [v for v in sorted_vals if _lt(v, current)]
                if tighter:
                    candidates.append((cspec.name, tighter[0]))

            elif op in (ConstraintOp.EQ, ConstraintOp.EQ_ANY):
                pass  # already maximally restrictive

            elif op == ConstraintOp.BOOLEAN:
                pass  # only one meaningful value

            elif op in (ConstraintOp.CONTAINS_ANY, ConstraintOp.CONTAINS):
                # Replace with a rarer tag value
                if pool_index:
                    rarest_val = None
                    rarest_size = float("inf")
                    for v in cspec.sampling_values:
                        if v == current:
                            continue
                        m = pool_index.get_mask(cspec.name, v)
                        sz = pool_index.pool_size(m)
                        if sz < rarest_size:
                            rarest_size = sz
                            rarest_val = v
                    if rarest_val is not None:
                        candidates.append((cspec.name, rarest_val))

            elif op == ConstraintOp.CONTAINS_ALL:
                # Add an additional required tag
                if isinstance(current, (list, tuple)):
                    for v in cspec.sampling_values:
                        if v not in current:
                            candidates.append((cspec.name, list(current) + [v]))
                            break

            elif op in (ConstraintOp.NOT_CONTAINS, ConstraintOp.NOT_CONTAINS_ANY):
                # Exclude the most common tag value
                if pool_index:
                    most_common_val = None
                    most_common_size = -1
                    full = pool_index.full_mask()
                    for v in cspec.sampling_values:
                        if v == current:
                            continue
                        m = pool_index.get_mask(cspec.name, v)
                        sz = pool_index.pool_size(m)
                        if sz > most_common_size:
                            most_common_size = sz
                            most_common_val = v
                    if most_common_val is not None:
                        candidates.append((cspec.name, most_common_val))

            elif op == ConstraintOp.IN_SET:
                # Narrow the allowed set
                if isinstance(current, (list, tuple)) and len(current) > 1:
                    candidates.append((cspec.name, [current[0]]))

        else:
            # Constraint not in original set: add with most restrictive value
            if op == ConstraintOp.LTE:
                candidates.append((cspec.name, sorted_vals[0]))
            elif op == ConstraintOp.GTE:
                candidates.append((cspec.name, sorted_vals[-1]))
            elif op == ConstraintOp.RANGE:
                candidates.append((cspec.name, sorted_vals[0]))
            elif op in (ConstraintOp.EQ, ConstraintOp.EQ_ANY, ConstraintOp.CONTAINS, ConstraintOp.CONTAINS_ANY):
                if pool_index:
                    rarest_val = None
                    rarest_size = float("inf")
                    for v in cspec.sampling_values:
                        m = pool_index.get_mask(cspec.name, v)
                        sz = pool_index.pool_size(m)
                        if sz < rarest_size:
                            rarest_size = sz
                            rarest_val = v
                    if rarest_val is not None:
                        candidates.append((cspec.name, rarest_val))
                else:
                    candidates.append((cspec.name, sorted_vals[-1]))
            elif op in (ConstraintOp.NOT_CONTAINS, ConstraintOp.NOT_CONTAINS_ANY):
                if pool_index:
                    most_common_val = None
                    most_common_size = -1
                    for v in cspec.sampling_values:
                        m = pool_index.get_mask(cspec.name, v)
                        sz = pool_index.pool_size(m)
                        if sz > most_common_size:
                            most_common_size = sz
                            most_common_val = v
                    if most_common_val is not None:
                        candidates.append((cspec.name, most_common_val))
                else:
                    candidates.append((cspec.name, sorted_vals[0]))
            elif op != ConstraintOp.BOOLEAN:
                candidates.append((cspec.name, sorted_vals[-1]))

    return candidates


# ---------------------------------------------------------------------------
# Soft preferences from unused constraints
# ---------------------------------------------------------------------------

_MIN_DISCRIMINATING_RATIO = 0.05
_MAX_DISCRIMINATING_RATIO = 0.95


def _direction_from_operator(op: ConstraintOp, soft_direction: str | None) -> str:
    """Derive preference direction from constraint operator."""
    if soft_direction:
        return soft_direction
    if op in (ConstraintOp.GTE,):
        return "maximize"
    if op in (ConstraintOp.LTE,):
        return "minimize"
    return "match"


def _has_pool_variance(
    pool: pd.DataFrame,
    attribute: str,
    direction: str,
    value: Any = None,
) -> bool:
    """Check whether a preference on this attribute would discriminate in the pool."""
    if attribute not in pool.columns:
        return False
    col = pool[attribute].dropna()
    if len(col) == 0:
        return False

    if direction in ("minimize", "maximize"):
        try:
            numeric = col.astype(float)
            return numeric.min() != numeric.max()
        except (ValueError, TypeError):
            return False

    # direction == "match": value must cover some-but-not-all products
    if value is None:
        return False
    if isinstance(value, bool):
        matches = col.astype(str).str.lower() == str(value).lower()
    elif isinstance(value, (list, tuple)):
        matches = col.apply(
            lambda x: any(str(v).lower() in str(x).lower() for v in value)
            if x is not None else False
        )
    else:
        matches = col.astype(str).str.lower() == str(value).lower()

    ratio = matches.sum() / len(pool)
    return _MIN_DISCRIMINATING_RATIO <= ratio <= _MAX_DISCRIMINATING_RATIO


def sample_unused_constraint_prefs(
    hard_constraints: dict[str, Any],
    pool: pd.DataFrame,
    config: DomainConfig,
    n_target: int = 2,
    diversity: "DiversityTracker | None" = None,
) -> list[dict]:
    """Sample soft preferences from constraints NOT used as hard constraints.

    Picks unused filterable constraints whose attributes have discriminating
    variance in the pool.  The pool and hard constraints are never modified.
    """
    used_names = set(hard_constraints.keys())
    candidates: list[tuple[ConstraintSpec, str, Any]] = []

    for cspec in config.get_filterable_constraints():
        if cspec.name in used_names:
            continue
        if not cspec.sampling_values:
            continue

        direction = _direction_from_operator(cspec.operator, cspec.soft_direction)

        if direction in ("minimize", "maximize"):
            if _has_pool_variance(pool, cspec.attribute, direction):
                candidates.append((cspec, direction, None))
        else:
            for val in cspec.sampling_values:
                if _has_pool_variance(pool, cspec.attribute, "match", val):
                    candidates.append((cspec, "match", val))
                    break

    if not candidates:
        return []

    n_pick = min(n_target, len(candidates))

    # Diversity-aware weighting: prefer underused constraint attributes
    seen_attrs: set[str] = set()
    weights: list[float] = []
    for cspec, direction, _val in candidates:
        w = 1.0
        if diversity:
            w = 1.0 / (1.0 + diversity.pref_pair_counts.get((cspec.attribute, direction), 0))
        weights.append(w)

    selected: list[tuple[ConstraintSpec, str, Any]] = []
    remaining = list(range(len(candidates)))

    for _i in range(n_pick):
        if not remaining:
            break
        cur_w = [weights[i] for i in remaining]
        total = sum(cur_w)
        if total <= 0:
            break
        pick = random.choices(range(len(remaining)), weights=cur_w, k=1)[0]
        idx = remaining.pop(pick)
        chosen_cspec, chosen_dir, chosen_val = candidates[idx]
        if chosen_cspec.attribute in seen_attrs:
            continue
        seen_attrs.add(chosen_cspec.attribute)
        selected.append((chosen_cspec, chosen_dir, chosen_val))

    prefs: list[dict] = []
    for cspec, direction, val in selected:
        pref: dict[str, Any] = {
            "attribute": cspec.attribute,
            "direction": direction,
            "from_unused_constraint": cspec.name,
        }
        if direction == "match" and val is not None:
            pref["target"] = val
        prefs.append(pref)

    return prefs


def _lt(a: Any, b: Any) -> bool:
    try:
        return float(a) < float(b)
    except (ValueError, TypeError):
        return False


def _gt(a: Any, b: Any) -> bool:
    try:
        return float(a) > float(b)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Description-based constraints
# ---------------------------------------------------------------------------

MAX_DESC_MATCH_RATIO = 0.40
MIN_DESC_MATCH_COUNT = 1


def _desc_product_matches(attr_name: str, required_val: Any, product_attrs: dict) -> bool:
    actual = product_attrs.get(attr_name)
    if actual is None:
        return False
    actual_str = str(actual).lower().strip()
    required_str = str(required_val).lower().strip()

    if required_str in ("true", "false"):
        return actual_str == required_str
    try:
        return float(actual) >= float(required_val)
    except (ValueError, TypeError):
        pass
    return actual_str == required_str


def filter_pool_by_desc(
    pool: pd.DataFrame,
    attr_name: str,
    attr_val: Any,
    desc_attrs: dict[str, dict],
    id_column: str = "product_name",
) -> pd.DataFrame:
    keep = [
        idx for idx, pid in pool[id_column].items()
        if _desc_product_matches(attr_name, attr_val, desc_attrs.get(pid, {}))
    ]
    return pool.loc[pool.index.isin(keep)]


def sample_desc_constraint(
    pool: pd.DataFrame,
    desc_attrs: dict[str, dict],
    difficulty: str,
    config: DomainConfig,
    desc_catalog: dict | None = None,
) -> Optional[tuple[str, Any, pd.DataFrame]]:
    """Sample a description-based constraint from the discovered attribute space."""
    bracket = config.difficulty.get(difficulty)
    prob = bracket.desc_constraint_probability if bracket else 0.3
    if random.random() > prob:
        return None
    if not desc_catalog:
        return None

    usable_attrs = desc_catalog.get("attributes", [])
    coverage_stats = desc_catalog.get("coverage_stats", {})
    if not usable_attrs:
        return None

    attr_pool = list(usable_attrs)
    random.shuffle(attr_pool)

    for attr_name in attr_pool:
        stats = coverage_stats.get(attr_name, {})
        top_values = stats.get("top_values", [])
        attr_type = stats.get("type", "categorical")

        if not top_values:
            continue

        viable = [tv["value"] for tv in top_values if tv.get("count", 0) >= 20]
        if not viable:
            continue

        random.shuffle(viable)
        for val in viable:
            if attr_type == "boolean":
                val = str(val).lower() == "true"

            filtered = filter_pool_by_desc(pool, attr_name, val, desc_attrs, config.id_column)
            n_match = len(filtered)
            if n_match < MIN_DESC_MATCH_COUNT:
                continue
            if n_match / max(len(pool), 1) > MAX_DESC_MATCH_RATIO:
                continue

            return f"desc_{attr_name}", val, filtered

    return None


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

def pick_anchor(
    pool: pd.DataFrame,
    config: DomainConfig,
    used_ids: set[str],
) -> Optional[pd.Series]:
    """Pick an anchor product from the pool, preferring complete + popular items."""
    if len(pool) == 0:
        return None

    id_col = config.id_column
    if id_col in pool.columns:
        eligible = pool[~pool[id_col].astype(str).isin(used_ids)]
    else:
        eligible = pool
    if len(eligible) == 0:
        eligible = pool

    optional_attrs = [
        a.name for a in config.attributes
        if not a.required and a.name in eligible.columns
    ]
    completeness = eligible[optional_attrs].notna().sum(axis=1).astype(float) if optional_attrs else pd.Series(1.0, index=eligible.index)

    pop_cols = [a.name for a in config.attributes if a.popularity_proxy and a.name in eligible.columns]
    if pop_cols:
        popularity = eligible[pop_cols[0]].fillna(1).apply(np.log1p).astype(float)
    else:
        popularity = pd.Series(1.0, index=eligible.index)

    weights = completeness * 2.0 + popularity * 1.0
    weights = weights.clip(lower=0.1)
    weights = weights / weights.sum()

    return eligible.sample(1, weights=weights).iloc[0]


# ---------------------------------------------------------------------------
# LLM profile generation
# ---------------------------------------------------------------------------

EXPERTISE_DISTRIBUTION = ["novice"] * 30 + ["intermediate"] * 40 + ["expert"] * 30


def _format_constraints_for_prompt(
    constraints: dict[str, Any],
    config: DomainConfig,
) -> str:
    lines = []
    for key, val in constraints.items():
        label = config.get_constraint_label(key, val)
        if label:
            lines.append(f"  - {key}: {val}  ({label})")
        elif key.startswith("desc_"):
            attr_name = key[5:]
            lines.append(f"  - {key}: {val}  (Description-based: {attr_name.replace('_', ' ')})")
        else:
            lines.append(f"  - {key}: {val}")
    lines.append("")
    lines.append(f"  Constraint keys (use EXACTLY these in the revelation plan): {list(constraints.keys())}")
    return "\n".join(lines)


def _format_products_for_prompt(
    products: pd.DataFrame,
    config: DomainConfig,
) -> str:
    display_attrs = [
        a for a in config.attributes
        if a.filterable or a.preference_eligible or a.popularity_proxy
    ]
    lines = []
    for i, (_, p) in enumerate(products.iterrows(), 1):
        specs = []
        for attr in display_attrs:
            val = p.get(attr.name)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                unit = f" {attr.unit}" if attr.unit else ""
                specs.append(f"{attr.display_name}: {val}{unit}")
        lines.append(f"  {config.item_noun.title()} {i}: {', '.join(specs[:8])}")
    return "\n".join(lines)


def _build_trigger_list(config: DomainConfig) -> dict[str, list[str]]:
    """Build trigger lists for the prompt from the config's trigger/violation specs."""
    responsive_triggers = [t.name for t in config.triggers]
    reactive_triggers = [vt.name for vt in config.violation_triggers]
    return {
        "responsive": responsive_triggers,
        "reactive": reactive_triggers,
    }


def _format_assigned_preferences(pairs: list[tuple[str, str]]) -> str:
    """Format pre-sampled (attribute, direction) pairs for the LLM prompt."""
    lines = []
    for i, (attr, direction) in enumerate(pairs, 1):
        lines.append(f'  {i}. attribute="{attr}", direction="{direction}"')
    return "\n".join(lines) if lines else "  (none assigned)"


def _collect_forbidden_terms(
    constraints: dict[str, Any],
    config: DomainConfig,
) -> list[str]:
    """Extract constraint values the initial_query must not contain."""
    terms: list[str] = []
    for key, val in constraints.items():
        if isinstance(val, bool) or val is None:
            continue
        cspec = config._constraint_by_name.get(key)
        if cspec and cspec.sampling_values and len(cspec.sampling_values) < _BROAD_VOCAB_THRESHOLD:
            continue
        if isinstance(val, str) and len(val) >= _MIN_LEAK_LEN:
            terms.append(val)
        elif isinstance(val, (list, tuple)):
            terms.extend(str(v) for v in val if isinstance(v, str) and len(v) >= _MIN_LEAK_LEN)
    return terms


def generate_profile(
    constraints: dict[str, Any],
    pool: pd.DataFrame,
    config: DomainConfig,
    difficulty: str,
    model: str = "gpt-4.1-mini",
    assigned_preferences: list[tuple[str, str]] | None = None,
    query_feedback: str | None = None,
) -> Optional[dict]:
    """Generate a user profile via LLM, grounded by constraints + sample products."""
    sample_n = min(3, len(pool))
    if sample_n == 0:
        return None
    sample_products = pool.sample(sample_n)

    expertise_level = random.choice(EXPERTISE_DISTRIBUTION)
    trigger_lists = _build_trigger_list(config)

    domain_desc = (config.prompt_fragments.domain_description or "").strip()
    if not domain_desc:
        raise ValueError(
            "prompt_fragments.domain_description is required for LLM profile generation. "
            "It is set automatically when using generate_domain_config() from a triage with "
            "domain_description, or set it in the DomainConfig / JSON.",
        )

    template = Path(resolve_prompt_path("generate_profile.txt")).read_text(
        encoding="utf-8"
    )
    forbidden = _collect_forbidden_terms(constraints, config)
    if forbidden:
        forbidden_text = (
            "FORBIDDEN TERMS in initial_query — your query will be rejected if it "
            "contains ANY of these words/phrases (case-insensitive). These are "
            "constraint values that must only be revealed later through conversation, "
            "NOT in the opening query:\n  "
            + ", ".join(f'"{t}"' for t in forbidden)
        )
    else:
        forbidden_text = ""

    user_prompt = format_prompt_template(
        template,
        domain_description=domain_desc,
        item_noun=config.item_noun,
        item_noun_plural=config.item_noun_plural,
        constraints_text=_format_constraints_for_prompt(constraints, config),
        products_text=_format_products_for_prompt(sample_products, config),
        difficulty=difficulty,
        expertise_level=expertise_level,
        responsive_triggers=json.dumps(trigger_lists["responsive"]),
        assigned_preferences=_format_assigned_preferences(
            assigned_preferences or [],
        ),
        query_rules=config.prompt_fragments.query_rules or "",
        forbidden_terms=forbidden_text,
        use_case_description=config.prompt_fragments.use_case_description or "",
        system_persona=config.prompt_fragments.system_persona
        or f"a {config.item_noun} recommendation assistant",
    )
    if query_feedback:
        user_prompt += f"\n\nIMPORTANT CORRECTION: {query_feedback}"
    try:
        profile = call_llm_json(
            user_prompt,
            model,
            temperature=0.9,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        return profile
    except Exception:
        logger.warning(
            "generate_profile: call_llm_json failed difficulty=%r model=%r",
            difficulty,
            model,
            exc_info=True,
        )
        return None


def build_revelation_plan(
    constraints: dict[str, Any],
    config: DomainConfig,
    difficulty: str,
) -> dict:
    """Deterministically construct a revelation plan from difficulty targets.

    Assigns each constraint to a category based on difficulty:
    - easy: 60%+ proactive, 0 reactive
    - medium: 30-50% proactive, 0-1 reactive
    - hard: <30% proactive, 2+ reactive
    """
    keys = list(constraints.keys())
    n = len(keys)
    if n == 0:
        return {"proactive": [], "responsive": {}, "reactive": {}, "contextual": {}}

    random.shuffle(keys)

    # Determine split ratios based on difficulty
    if difficulty == "small":
        n_proactive = max(1, int(n * 0.6) + (1 if n > 1 else 0))
        n_reactive = 0
    elif difficulty == "large":
        n_proactive = max(1, min(int(n * 0.25), n - 2))
        n_reactive = min(2, n - n_proactive)
    elif difficulty in ("oc_feasible", "oc_infeasible"):
        n_proactive = max(1, int(n * 0.3))
        n_reactive = min(1, n - n_proactive)
    else:  # medium
        n_proactive = max(1, int(n * 0.4))
        n_reactive = min(1, n - n_proactive)

    n_responsive = n - n_proactive - n_reactive

    proactive_keys = keys[:n_proactive]
    responsive_keys = keys[n_proactive:n_proactive + n_responsive]
    reactive_keys = keys[n_proactive + n_responsive:]

    # Build responsive: map constraints to triggers
    responsive: dict[str, list[str]] = {}
    trigger_map = _build_constraint_to_trigger_map(config)
    for ckey in responsive_keys:
        trigger = trigger_map.get(ckey)
        if trigger:
            responsive.setdefault(trigger, []).append(ckey)
        else:
            # Fallback: use first available trigger or generic name
            fallback = config.triggers[0].name if config.triggers else "agent_asks_details"
            responsive.setdefault(fallback, []).append(ckey)

    # Build reactive: map constraints to violation triggers
    reactive: dict[str, list[str]] = {}
    vt_map = {vt.constraint_name: vt.name for vt in config.violation_triggers}
    for ckey in reactive_keys:
        vt_name = vt_map.get(ckey, f"shown_wrong_{ckey}")
        reactive[vt_name] = [ckey]

    return {
        "proactive": proactive_keys,
        "responsive": responsive,
        "reactive": reactive,
        "contextual": {},
    }


def _build_constraint_to_trigger_map(config: DomainConfig) -> dict[str, str]:
    """Map each constraint name to its best-matching responsive trigger."""
    cmap: dict[str, str] = {}
    for trigger in config.triggers:
        for cname in trigger.unlocks_constraints:
            if cname not in cmap:
                cmap[cname] = trigger.name
    return cmap


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_revelation_plan(
    hard_constraints: dict[str, Any],
    revelation_plan: dict,
) -> tuple[bool, str]:
    """Every hard constraint must appear in exactly one revelation category."""
    if not isinstance(revelation_plan, dict):
        return False, f"Revelation plan is not a dict: {type(revelation_plan)}"

    all_keys = set(hard_constraints.keys())
    assigned: set[str] = set()

    for key in revelation_plan.get("proactive", []):
        if isinstance(key, str):
            assigned.add(key)

    for _trigger, keys in revelation_plan.get("responsive", {}).items():
        if isinstance(keys, list):
            assigned.update(k for k in keys if isinstance(k, str))
        elif isinstance(keys, str):
            assigned.add(keys)

    for _trigger, keys in revelation_plan.get("reactive", {}).items():
        if isinstance(keys, list):
            assigned.update(k for k in keys if isinstance(k, str))
        elif isinstance(keys, str):
            assigned.add(keys)

    for _trigger, info in revelation_plan.get("contextual", {}).items():
        if isinstance(info, dict):
            reveals = info.get("reveals", [])
            if isinstance(reveals, list):
                assigned.update(k for k in reveals if isinstance(k, str))
        elif isinstance(info, list):
            assigned.update(k for k in info if isinstance(k, str))

    missing = all_keys - assigned
    if missing:
        return False, f"Constraints not in any revelation category: {missing}"

    return True, "OK"


_NUMERIC_LEAK_RE = re.compile(
    r'\$\s?[\d,]+(?:\.\d+)?'    # prices: $30, $1,200.00
    r'|\d+\s*(?:'
    r'[Gg][Bb]|[Tt][Bb]'        # storage: 16GB, 1TB
    r'|[Kk][Gg]|lbs?'           # weight: 1.5kg, 3lbs
    r'|[Mm][Ll]|[Ff][Ll]\.?\s?[Oo][Zz]|[Oo][Zz]'  # volume: 50ml, 1.7 fl oz
    r'|[Hh][Pp]|[Mm][Pp][Gg]'   # power/efficiency: 300hp, 30mpg
    r'|hours?|hrs?'              # duration: 8 hours
    r'|["″"]'                   # screen: 15"
    r')',
)

_MIN_LEAK_LEN = 3
_BROAD_VOCAB_THRESHOLD = 8


def validate_query(
    initial_query: str,
    config: DomainConfig,
    constraints: dict[str, Any],
) -> tuple[bool, str]:
    """Ensure the query is vague and doesn't leak constraint values.

    Checks:
    1. No numeric values with units (prices, sizes, durations, etc.)
    2. No leaked string constraint values (brand names, ingredient terms,
       etc.) derived dynamically from the sampled constraints for this
       task.  Broad-vocabulary constraints (fewer than 8 sampling values,
       e.g. top-level categories) are skipped since their values are
       everyday words a real user would naturally use.
    3. Word count within [3, 15].
    """
    if _NUMERIC_LEAK_RE.search(initial_query):
        return False, f"Query leaks a numeric value: {_NUMERIC_LEAK_RE.search(initial_query).group()!r}"

    query_lower = initial_query.lower()
    for key, val in constraints.items():
        if isinstance(val, bool) or val is None:
            continue

        cspec = config._constraint_by_name.get(key)
        if cspec and cspec.sampling_values and len(cspec.sampling_values) < _BROAD_VOCAB_THRESHOLD:
            continue

        leak_terms: list[str] = []
        if isinstance(val, str):
            leak_terms.append(val)
        elif isinstance(val, (list, tuple)):
            leak_terms.extend(str(v) for v in val if isinstance(v, str))
        for term in leak_terms:
            if len(term) < _MIN_LEAK_LEN:
                continue
            if term.lower() in query_lower:
                return False, f"Query leaks constraint {key}={term!r}"

    word_count = len(initial_query.split())
    if word_count < 3 or word_count > 15:
        return False, f"Query word count {word_count} outside [3, 15]"

    return True, "OK"


def validate_attribute_preferences(
    attribute_preferences: list[dict],
    config: DomainConfig,
) -> tuple[bool, str]:
    """Validate LLM-generated attribute preferences against config.

    Allows 2-10 preferences.  Preferences sourced from unused constraints
    (those with ``from_unused_constraint``) skip the preference-attribute
    registry check since they originate from constraint specs.
    """
    if not isinstance(attribute_preferences, list):
        return False, "attribute_preferences is not a list"
    if len(attribute_preferences) < 2 or len(attribute_preferences) > 10:
        return False, f"Need 2-10 preferences, got {len(attribute_preferences)}"

    valid_attrs = {pa.attribute: pa for pa in config.preference_attributes}
    priorities: set[int] = set()

    for pref in attribute_preferences:
        if not isinstance(pref, dict):
            return False, "preference item is not a dict"

        is_from_unused = bool(pref.get("from_unused_constraint"))
        attr = pref.get("attribute")

        if not is_from_unused:
            if attr not in valid_attrs:
                return False, f"Invalid preference attribute: {attr}"
            direction = pref.get("direction")
            pa = valid_attrs[attr]
            if direction not in pa.directions:
                return False, f"Invalid direction '{direction}' for {attr}"
        else:
            direction = pref.get("direction")
            if direction not in ("minimize", "maximize", "match"):
                return False, f"Invalid direction '{direction}' for unused-constraint pref {attr}"

        if direction == "match" and "target" not in pref:
            return False, f"match direction requires 'target' for {attr}"

        priority = pref.get("priority")
        max_priority = len(attribute_preferences)
        if not isinstance(priority, int) or priority < 1 or priority > max_priority:
            return False, f"Invalid priority: {priority} (max {max_priority})"
        if priority in priorities:
            return False, f"Duplicate priority: {priority}"
        priorities.add(priority)

        mode = pref.get("revelation_mode")
        if mode not in ("responsive", "contextual"):
            return False, f"Invalid revelation_mode for preference: {mode}"

        if not pref.get("trigger"):
            return False, "Preference missing trigger"

        if mode == "contextual" and not pref.get("user_says"):
            return False, "Contextual preference missing user_says"

    return True, "OK"


# ---------------------------------------------------------------------------
# Difficulty assignment
# ---------------------------------------------------------------------------

def assign_difficulty(
    hard_constraints: dict[str, Any],
    revelation_plan: dict,
    pool_size: int,
    config: DomainConfig,
    *,
    infeasible: bool = False,
) -> str:
    """Assign difficulty from revelation plan structure + pool size.

    Falls back to bracket matching when the revelation-plan heuristics
    don't produce a clear answer.
    """
    if infeasible:
        return "oc_infeasible"
    if pool_size <= 3:
        return "oc_feasible"

    proactive = revelation_plan.get("proactive", [])
    n_proactive = len(proactive)
    n_total = len(hard_constraints)
    elicitation_ratio = 1.0 - (n_proactive / n_total) if n_total > 0 else 1.0

    n_reactive = len(revelation_plan.get("reactive", {}))

    hard_bracket = config.difficulty.get("large")
    if hard_bracket and n_reactive >= 2 and pool_size <= hard_bracket.pool_range[1] and n_total >= hard_bracket.min_constraints:
        return "large"

    if elicitation_ratio < 0.40 and n_reactive == 0:
        return "small"

    return "medium"


# ---------------------------------------------------------------------------
# Utility scoring
# ---------------------------------------------------------------------------

def score_pool_by_preferences(
    pool: pd.DataFrame,
    attribute_preferences: list[dict],
) -> list[tuple[str, float]]:
    """Score and rank products by preference utility. Returns best-first."""
    weights = compute_preference_weights(attribute_preferences)
    pool_stats = compute_pool_stats(pool, attribute_preferences)

    scored: list[tuple[str, float]] = []
    for pid, row in pool.iterrows():
        utility = compute_product_utility(row, attribute_preferences, pool_stats, weights)
        scored.append((pid, utility))

    scored.sort(key=lambda x: -x[1])
    return scored


# ---------------------------------------------------------------------------
# Build final task object
# ---------------------------------------------------------------------------

def build_task(
    anchor: pd.Series,
    task_constraints: dict[str, Any],
    profile: dict,
    difficulty: str,
    pool: pd.DataFrame,
    config: DomainConfig,
    task_id: str | None = None,
    desc_pool: pd.DataFrame | None = None,
    *,
    infeasible: bool = False,
    base_pool_size: int | None = None,
) -> dict:
    """Assemble the final task dictionary.

    Reward: ``I(constraints_ok) * (0.5 + 0.5 * preference_utility)``
    """
    attribute_preferences = profile.get("attribute_preferences", [])
    desc_keys = [k for k in task_constraints if k.startswith("desc_")]
    effective_pool = desc_pool if desc_pool is not None else pool

    if infeasible:
        weights = compute_preference_weights(attribute_preferences)
        target_set = {
            "ranking_method": "infeasible",
            "best_product_ids": [],
            "best_utility": 0.0,
            "preference_weights": [round(w, 4) for w in weights],
            "pool_stats": {},
        }
        best_utility = 0.0
        best_product_ids: list[str] = []
    else:
        scored = score_pool_by_preferences(effective_pool, attribute_preferences)
        best_utility = scored[0][1] if scored else 0.0
        best_product_ids = [pid for pid, u in scored if u == best_utility]

        weights = compute_preference_weights(attribute_preferences)
        pool_stats = compute_pool_stats(effective_pool, attribute_preferences)

        target_set = {
            "ranking_method": "continuous_weighted_utility",
            "best_product_ids": best_product_ids,
            "best_utility": round(best_utility, 4),
            "preference_weights": [round(w, 4) for w in weights],
            "pool_stats": {
                attr: {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}
                for attr, stats in pool_stats.items()
            },
        }

    revelation_plan = profile.get("constraint_revelation_plan", {})
    id_col = config.id_column

    task = {
        "task_id": task_id or "temporary_id",
        "domain": config.name,
        "difficulty": difficulty,
        "reward_type": "utility",
        "initial_query": profile["initial_query"],
        "user_profile": {
            "expertise": profile.get("expertise", "intermediate"),
            "hard_constraints": task_constraints,
            "use_case": profile.get("use_case", "general use"),
            "attribute_preferences": attribute_preferences,
            "constraint_revelation_plan": revelation_plan,
            "behavioral_profile": profile.get("behavioral_profile", {
                "patience_turns": 12,
                "response_verbosity": "brief",
            }),
        },
        "target_product_set": target_set,
        "metadata": {
            "anchor_product_id": anchor.get(id_col, ""),
            "structured_pool_size": len(pool),
            "desc_pool_size": len(desc_pool) if desc_pool is not None else None,
            "filtered_pool_size": len(effective_pool),
            "best_utility": round(best_utility, 4),
            "n_tied_best": len(best_product_ids),
            "n_hard_constraints": len(task_constraints),
            "n_unused_constraint_prefs": sum(1 for p in attribute_preferences if p.get("from_unused_constraint")),
            "n_attribute_preferences": len(attribute_preferences),
            "n_proactive": len(revelation_plan.get("proactive", [])),
            "n_responsive": len(revelation_plan.get("responsive", {})),
            "n_reactive": len(revelation_plan.get("reactive", {})),
            "desc_constraints": desc_keys,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    if infeasible:
        task["infeasible"] = True
        task["metadata"]["base_pool_size"] = base_pool_size

    return task


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_all_tasks(
    catalog: pd.DataFrame,
    config: DomainConfig,
    *,
    target_counts: dict[str, int] | None = None,
    model: str = "gpt-4.1-mini",
    seed: int = 42,
    max_attempts_multiplier: int = 12,
    desc_attrs: dict[str, dict] | None = None,
    desc_catalog: dict | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 10,
    value_frequencies: dict[str, dict[Any, float]] | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Generate benchmark tasks for a domain.

    Args:
        catalog: Cleaned catalog DataFrame.
        config: ``DomainConfig`` for this domain.
        target_counts: ``{difficulty: count}`` overrides config defaults.
        model: LLM model identifier for profile generation.
        seed: Random seed.
        max_attempts_multiplier: ``total_target * this`` = max attempts.
        desc_attrs: ``{product_id: {attr: value}}`` from description extraction.
        desc_catalog: Metadata about description attributes.
        checkpoint_path: Path for intermediate checkpointing. If the file
            exists when generation starts, tasks and loop state are restored
            and generation continues toward the same targets.
        checkpoint_every: Save a checkpoint after this many new accepted tasks
            (in addition to a final save when the path is set).
        value_frequencies: Precomputed ``{constraint: {value: hit_rate}}``
            from :func:`catalog_hygiene.compute_value_frequencies`.  When
            provided, enables frequency-weighted value sampling.
        max_workers: Number of parallel threads for LLM profile generation.
            Higher values speed up generation at the cost of slightly
            stale diversity tracking within each batch.

    Returns:
        List of task dicts.
    """
    random.seed(seed)

    if target_counts is None:
        target_counts = {name: bracket.target_count for name, bracket in config.difficulty.items()}

    total_target = sum(target_counts.values())
    max_attempts = total_target * max_attempts_multiplier
    desc_attrs = desc_attrs or {}

    # Compute value frequencies on the fly if not provided
    if value_frequencies is None:
        from generation.catalog_hygiene import compute_value_frequencies
        value_frequencies = compute_value_frequencies(catalog, config)

    # Build bitmask pool index for guided constraint sampling
    logger.info("Building pool index (bitmask) over %d catalog rows...", len(catalog))
    pool_index = PoolIndex(catalog, config)
    logger.info("Pool index built: %d constraint-value pairs indexed",
                len(pool_index._index))

    diversity = DiversityTracker(config)

    rejections_template = {
        "constraint_sampling": 0,
        "llm_failure": 0,
        "revelation_plan": 0,
        "query_invalid": 0,
        "query_duplicate": 0,
        "preference_invalid": 0,
        "no_anchor": 0,
    }

    tasks: list[dict] = []
    used_ids: set[str] = set()
    seen_queries: set[str] = set()
    attempts = 0
    rejections = dict(rejections_template)

    if checkpoint_path:
        loaded = _load_checkpoint(checkpoint_path)
        if loaded:
            tasks = list(loaded.get("tasks") or [])
            used_ids = {str(x) for x in (loaded.get("used_product_ids") or []) if str(x)}
            raw_sq = loaded.get("seen_queries")
            if isinstance(raw_sq, list):
                seen_queries = {str(s).lower().strip() for s in raw_sq if str(s).strip()}
            elif tasks:
                seen_queries = {
                    str(t.get("initial_query", "")).lower().strip()
                    for t in tasks
                    if t.get("initial_query")
                }
            attempts = int(loaded.get("attempts", 0))
            rj = loaded.get("rejections")
            if isinstance(rj, dict):
                for k in rejections:
                    if k in rj:
                        try:
                            rejections[k] = int(rj[k])
                        except (TypeError, ValueError):
                            pass
            for t in tasks:
                diversity.record(
                    t.get("user_profile", {}).get("hard_constraints", {}),
                    t.get("user_profile", {}).get("attribute_preferences"),
                )
            logger.info(
                "Resumed from checkpoint %s: %d tasks, attempts=%d, saved_at=%s",
                checkpoint_path,
                len(tasks),
                attempts,
                loaded.get("saved_at", "?"),
            )

    def counts_by_diff() -> dict[str, int]:
        c: dict[str, int] = {d: 0 for d in target_counts}
        for t in tasks:
            d = t.get("difficulty", "unknown")
            if d in c:
                c[d] += 1
        return c

    def tier_needed(d: str) -> bool:
        return counts_by_diff().get(d, 0) < target_counts.get(d, 0)

    def targets_met() -> bool:
        c = counts_by_diff()
        return all(c.get(d, 0) >= target_counts.get(d, 0) for d in target_counts)

    logger.info("Starting task generation for domain '%s'", config.name)
    logger.info("Target: %d tasks (%s)", total_target, target_counts)
    logger.info(
        "Catalog: %d rows | id=%s | max_attempts=%d (total_target × %d)",
        len(catalog), config.id_column, max_attempts, max_attempts_multiplier,
    )

    last_query_feedback: dict[str, str] = {}

    _executor = ThreadPoolExecutor(max_workers=max_workers)
    _in_flight: dict = {}

    while not targets_met() and attempts < max_attempts:
        # Fill pipeline up to max_workers
        while len(_in_flight) < max_workers and not targets_met() and attempts < max_attempts:
            attempts += 1

            if attempts == 1 or attempts % 5 == 0 or attempts == max_attempts:
                c = counts_by_diff()
                short_now = {d: max(0, target_counts.get(d, 0) - c.get(d, 0)) for d in target_counts}
                logger.info(
                    "Progress: attempt %d/%d | done %d/%d | by_tier %s | still_need %s | rj %s",
                    attempts, max_attempts,
                    len(tasks), total_target,
                    c, short_now,
                    {k: v for k, v in rejections.items() if v},
                )

            needed = [d for d in target_counts if tier_needed(d)]
            if not needed:
                break
            counts = counts_by_diff()
            shortfalls = [max(0, target_counts[d] - counts.get(d, 0)) for d in needed]
            total_short = sum(shortfalls)
            if total_short == 0:
                break
            weights_dist = [s / total_short for s in shortfalls]
            target_difficulty = random.choices(needed, weights=weights_dist, k=1)[0]

            is_oc = target_difficulty in ("oc_feasible", "oc_infeasible")
            is_infeasible = target_difficulty == "oc_infeasible"
            base_pool = None

            if is_oc:
                result = sample_oc_constraint_set(
                    catalog, config, infeasible=is_infeasible,
                    diversity=diversity, value_frequencies=value_frequencies,
                    pool_index=pool_index,
                )
                if result is not None:
                    filter_constraints, pool, base_pool = result
                else:
                    rejections["constraint_sampling"] += 1
                    continue
            else:
                result = sample_constraint_set(
                    catalog, config, target_difficulty,
                    diversity=diversity, value_frequencies=value_frequencies,
                    pool_index=pool_index,
                )
                if result is None:
                    rejections["constraint_sampling"] += 1
                    continue
                filter_constraints, pool = result

            # Optional description constraint
            desc_pool = None
            if desc_attrs and not is_infeasible:
                desc_result = sample_desc_constraint(pool, desc_attrs, target_difficulty, config, desc_catalog)
                if desc_result is not None:
                    desc_key, desc_val, desc_pool = desc_result
                    filter_constraints[desc_key] = desc_val

            # Anchor selection
            if is_infeasible:
                anchor_source = base_pool
            elif desc_pool is not None:
                anchor_source = desc_pool
            else:
                anchor_source = pool

            anchor = pick_anchor(anchor_source, config, used_ids)
            if anchor is None:
                rejections["no_anchor"] += 1
                na = rejections["no_anchor"]
                if na <= 3 or na % 25 == 0:
                    logger.info(
                        "No anchor (count=%d): attempt %d tier=%s pool_len=%d base_pool_len=%s",
                        na, attempts, target_difficulty, len(anchor_source),
                        None if base_pool is None else len(base_pool),
                    )
                continue

            # --- Soft preferences from unused constraints ---
            assigned_prefs_base = diversity.sample_preference_pairs(config)
            unused_prefs: list[dict] = []
            if not is_oc:
                pref_pool = pool
                unused_prefs = sample_unused_constraint_prefs(
                    filter_constraints, pref_pool, config,
                    n_target=2, diversity=diversity,
                )

            profile_pool = base_pool if is_infeasible else pool
            assigned_prefs = list(assigned_prefs_base)

            # Submit LLM call to thread pool
            future = _executor.submit(
                generate_profile,
                filter_constraints, profile_pool, config, target_difficulty,
                model=model, assigned_preferences=assigned_prefs,
                query_feedback=last_query_feedback.get(target_difficulty),
            )
            _in_flight[future] = {
                'filter_constraints': filter_constraints,
                'target_difficulty': target_difficulty,
                'unused_prefs': unused_prefs,
                'anchor': anchor,
                'pool': pool,
                'desc_pool': desc_pool,
                'base_pool': base_pool,
                'is_oc': is_oc,
                'is_infeasible': is_infeasible,
            }

        if not _in_flight:
            break

        # Wait for at least one LLM call to complete
        done, _ = wait(set(_in_flight.keys()), return_when=FIRST_COMPLETED)

        for future in done:
            cand = _in_flight.pop(future)
            try:
                profile = future.result()
            except Exception:
                logger.warning(
                    "Task gen: LLM worker future failed (llm_failure)",
                    exc_info=True,
                )
                rejections["llm_failure"] += 1
                continue

            if profile is None:
                logger.warning(
                    "Task gen: generate_profile returned None (llm_failure) — "
                    "see prior shared.llm / generate_profile warnings for cause",
                )
                rejections["llm_failure"] += 1
                continue

            target_difficulty = cand['target_difficulty']
            filter_constraints = cand['filter_constraints']
            unused_prefs = cand['unused_prefs']
            is_infeasible = cand['is_infeasible']
            pool = cand['pool']
            desc_pool = cand['desc_pool']
            base_pool = cand['base_pool']
            anchor = cand['anchor']

            # Merge unused-constraint preferences into the profile
            if unused_prefs:
                llm_prefs = profile.get("attribute_preferences", [])
                next_priority = max((p.get("priority", 0) for p in llm_prefs), default=0) + 1
                responsive_triggers = [t.name for t in config.triggers]
                for up in unused_prefs:
                    trigger = responsive_triggers[0] if responsive_triggers else "agent_asks_general"
                    cname = up.get("from_unused_constraint", "")
                    cspec = config._constraint_by_name.get(cname)
                    if cspec:
                        for t in config.triggers:
                            if cspec.name in t.unlocks_constraints:
                                trigger = t.name
                                break
                    up["priority"] = next_priority
                    up["revelation_mode"] = "responsive"
                    up["trigger"] = trigger
                    up["description"] = f"Soft preference (from unused constraint {cname})"
                    next_priority += 1
                    llm_prefs.append(up)
                profile["attribute_preferences"] = llm_prefs

            if "initial_query" not in profile:
                keys = (
                    list(profile.keys()) if isinstance(profile, dict) else f"type={type(profile)}"
                )
                nested_hint = None
                if isinstance(profile, dict):
                    for label, cand in (
                        ("user_profile", profile.get("user_profile")),
                        ("profile", profile.get("profile")),
                        ("task", profile.get("task")),
                    ):
                        if isinstance(cand, dict) and "initial_query" in cand:
                            nested_hint = f"initial_query present under {label!r}"
                            break
                logger.warning(
                    "Task gen: profile JSON missing top-level 'initial_query' (llm_failure) "
                    "keys=%s nested_hint=%r",
                    keys,
                    nested_hint,
                )
                rejections["llm_failure"] += 1
                continue

            # Validate preferences
            attr_prefs = profile.get("attribute_preferences", [])
            valid, reason = validate_attribute_preferences(attr_prefs, config)
            if not valid:
                rejections["preference_invalid"] += 1
                logger.debug("Preferences invalid: %s", reason)
                continue

            # Validate query
            valid, reason = validate_query(
                profile["initial_query"], config, filter_constraints,
            )
            if not valid:
                rejections["query_invalid"] += 1
                logger.debug("Query rejected: %s | query=%r", reason, profile["initial_query"])
                last_query_feedback[target_difficulty] = (
                    f"Your previous initial_query was rejected: {reason}. "
                    "Rephrase to avoid this issue."
                )
                continue

            # Dedup
            q_key = profile["initial_query"].lower().strip()
            if q_key in seen_queries:
                rejections["query_duplicate"] += 1
                continue

            # Build revelation plan deterministically
            revelation_plan = build_revelation_plan(
                filter_constraints, config, target_difficulty,
            )
            valid, reason = validate_revelation_plan(filter_constraints, revelation_plan)
            if not valid:
                rejections["revelation_plan"] += 1
                logger.debug("Revelation plan invalid: %s", reason)
                continue

            difficulty = assign_difficulty(
                filter_constraints, revelation_plan, len(pool), config,
                infeasible=is_infeasible,
            )

            if not tier_needed(difficulty):
                continue

            profile["constraint_revelation_plan"] = revelation_plan

            # Build task
            task = build_task(
                anchor, filter_constraints, profile, difficulty,
                pool, config, task_id=f"task_{len(tasks):04d}",
                desc_pool=desc_pool,
                infeasible=is_infeasible,
                base_pool_size=len(base_pool) if base_pool is not None else None,
            )

            tasks.append(task)
            aid = anchor.get(config.id_column, "")
            if aid is not None and aid != "":
                used_ids.add(str(aid))
            seen_queries.add(q_key)
            diversity.record(filter_constraints, attr_prefs)
            last_query_feedback.pop(target_difficulty, None)

            counts = counts_by_diff()
            logger.info(
                "Accepted task %d/%d [%s] | %s | attempts=%d accept=%.0f%%",
                len(tasks), total_target, difficulty,
                " ".join(f"{d}={counts.get(d, 0)}" for d in target_counts),
                attempts,
                len(tasks) / max(attempts, 1) * 100,
            )

            if (
                checkpoint_path
                and checkpoint_every > 0
                and len(tasks) % checkpoint_every == 0
            ):
                _save_checkpoint(
                    checkpoint_path, tasks, used_ids, seen_queries,
                    attempts, rejections,
                )

    # Drain remaining in-flight futures (LLM calls already paid for)
    if _in_flight:
        for future in as_completed(_in_flight):
            cand = _in_flight[future]
            try:
                profile = future.result()
            except Exception:
                continue
            if profile is None:
                continue

            target_difficulty = cand['target_difficulty']
            filter_constraints = cand['filter_constraints']
            unused_prefs = cand['unused_prefs']
            is_infeasible = cand['is_infeasible']
            pool = cand['pool']
            desc_pool = cand['desc_pool']
            base_pool = cand['base_pool']
            anchor = cand['anchor']

            if unused_prefs:
                llm_prefs = profile.get("attribute_preferences", [])
                next_priority = max((p.get("priority", 0) for p in llm_prefs), default=0) + 1
                responsive_triggers = [t.name for t in config.triggers]
                for up in unused_prefs:
                    trigger = responsive_triggers[0] if responsive_triggers else "agent_asks_general"
                    cname = up.get("from_unused_constraint", "")
                    cspec = config._constraint_by_name.get(cname)
                    if cspec:
                        for t in config.triggers:
                            if cspec.name in t.unlocks_constraints:
                                trigger = t.name
                                break
                    up["priority"] = next_priority
                    up["revelation_mode"] = "responsive"
                    up["trigger"] = trigger
                    up["description"] = f"Soft preference (from unused constraint {cname})"
                    next_priority += 1
                    llm_prefs.append(up)
                profile["attribute_preferences"] = llm_prefs

            if "initial_query" not in profile:
                continue
            attr_prefs = profile.get("attribute_preferences", [])
            valid, _ = validate_attribute_preferences(attr_prefs, config)
            if not valid:
                continue
            valid, _ = validate_query(profile["initial_query"], config, filter_constraints)
            if not valid:
                continue
            q_key = profile["initial_query"].lower().strip()
            if q_key in seen_queries:
                continue

            revelation_plan = build_revelation_plan(filter_constraints, config, target_difficulty)
            valid, _ = validate_revelation_plan(filter_constraints, revelation_plan)
            if not valid:
                continue
            difficulty = assign_difficulty(
                filter_constraints, revelation_plan, len(pool), config,
                infeasible=is_infeasible,
            )
            if not tier_needed(difficulty):
                continue

            profile["constraint_revelation_plan"] = revelation_plan
            task = build_task(
                anchor, filter_constraints, profile, difficulty,
                pool, config, task_id=f"task_{len(tasks):04d}",
                desc_pool=desc_pool,
                infeasible=is_infeasible,
                base_pool_size=len(base_pool) if base_pool is not None else None,
            )
            tasks.append(task)
            aid = anchor.get(config.id_column, "")
            if aid is not None and aid != "":
                used_ids.add(str(aid))
            seen_queries.add(q_key)
            diversity.record(filter_constraints, attr_prefs)

            counts = counts_by_diff()
            logger.info(
                "Accepted task %d/%d [%s] (from drain) | %s | attempts=%d",
                len(tasks), total_target, difficulty,
                " ".join(f"{d}={counts.get(d, 0)}" for d in target_counts),
                attempts,
            )

    _executor.shutdown(wait=True)


    if checkpoint_path and tasks:
        _save_checkpoint(
            checkpoint_path, tasks, used_ids, seen_queries,
            attempts, rejections,
        )

    if targets_met():
        logger.info("All target tiers satisfied after %d attempts.", attempts)
    else:
        c = counts_by_diff()
        short = {d: max(0, target_counts.get(d, 0) - c.get(d, 0)) for d in target_counts}
        if attempts >= max_attempts:
            logger.warning(
                "Hit max_attempts=%d. Short by tier: %s | rejections=%s",
                max_attempts, short, rejections,
            )
        else:
            logger.warning(
                "Stopped before targets (attempts=%d). Short by tier: %s | rejections=%s",
                attempts, short, rejections,
            )

    logger.info(
        "Generation complete: %d tasks from %d attempts (%.1f%% acceptance)",
        len(tasks), attempts, len(tasks) / max(attempts, 1) * 100,
    )
    logger.info("Rejections: %s", rejections)

    # Log diversity stats
    if diversity.total_tasks > 0:
        used_constraints = {k: v for k, v in diversity.constraint_counts.items() if v > 0}
        total_constraints = len(diversity.constraint_counts)
        logger.info(
            "Diversity: %d/%d constraints used, value coverage: %s",
            len(used_constraints), total_constraints,
            {k: f"{sum(1 for c in v.values() if c > 0)}/{len(v)}"
             for k, v in diversity.value_counts.items() if any(c > 0 for c in v.values())},
        )
        n_unique_combos = len(diversity.pref_combo_counts)
        logger.info(
            "Preference diversity: %d unique combos across %d tasks, pair usage: %s",
            n_unique_combos, diversity.total_tasks,
            {f"{a}:{d}": cnt for (a, d), cnt in sorted(
                diversity.pref_pair_counts.items(), key=lambda x: -x[1]
            )},
        )

    return tasks


def _load_checkpoint(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load checkpoint %s: %s", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def _save_checkpoint(
    path: str,
    tasks: list[dict],
    used_ids: set[str],
    seen_queries: set[str],
    attempts: int,
    rejections: dict[str, int],
) -> None:
    data = {
        "tasks": tasks,
        "used_product_ids": list(used_ids),
        "seen_queries": list(seen_queries),
        "attempts": attempts,
        "rejections": dict(rejections),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=None, default=str))


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_generation_summary(tasks: list[dict], config: DomainConfig) -> None:
    """Print a human-readable summary of generated tasks."""
    counts: dict[str, int] = {}
    for t in tasks:
        d = t.get("difficulty", "unknown")
        counts[d] = counts.get(d, 0) + 1

    print("\n" + "=" * 60)
    print(f"  TASK GENERATION SUMMARY — {config.name}")
    print("=" * 60)
    print(f"\n  Total tasks: {len(tasks)}")

    for d_name, bracket in config.difficulty.items():
        actual = counts.get(d_name, 0)
        target = bracket.target_count
        status = "OK" if actual >= target else "INCOMPLETE"
        print(f"  {d_name:20s}: {actual:4d} / {target:4d}  [{status}]")

    if tasks:
        print(f"\n  Pool size distribution by difficulty:")
        for d_name in config.difficulty:
            tier = [t for t in tasks if t["difficulty"] == d_name]
            if tier:
                pools = [t["metadata"]["filtered_pool_size"] for t in tier]
                print(f"  {d_name:20s}: min={min(pools):3d}  "
                      f"median={int(np.median(pools)):3d}  "
                      f"max={max(pools):3d}")

        n_constraints = [t["metadata"]["n_hard_constraints"] for t in tasks]
        print(f"\n  Constraints per task: min={min(n_constraints)}, "
              f"max={max(n_constraints)}, mean={np.mean(n_constraints):.1f}")

        n_from_unused = [t["metadata"].get("n_unused_constraint_prefs", 0) for t in tasks]
        n_prefs = [t["metadata"]["n_attribute_preferences"] for t in tasks]
        tasks_with_unused = sum(1 for u in n_from_unused if u > 0)
        if tasks_with_unused > 0:
            print(f"\n  Unused-constraint prefs: {tasks_with_unused}/{len(tasks)} tasks")
            print(f"    From unused per task: min={min(n_from_unused)}, max={max(n_from_unused)}, mean={np.mean(n_from_unused):.1f}")
        print(f"  Total prefs per task: min={min(n_prefs)}, max={max(n_prefs)}, mean={np.mean(n_prefs):.1f}")

        desc_tasks = [t for t in tasks if t["metadata"].get("desc_constraints")]
        print(f"\n  Tasks with description constraints: {len(desc_tasks)}/{len(tasks)}")

    print("\n" + "=" * 60)
