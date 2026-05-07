"""Calibration sweep: empirically tune difficulty brackets and sampling probabilities.

Run ``calibration_sweep`` on a catalog + domain config to produce pool-size
distributions, then ``auto_tune_difficulty`` to adjust bracket ranges.
Iterate until distributions support all five difficulty levels.

``discover_constraint_counts`` runs an empirical sweep over constraint counts
to find the right ``min_constraints`` per difficulty tier for a given set of
target pool sizes.  This is useful for large catalogs where the default
fixed counts (3/4/5) produce pools that are far too large.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shared.config import DomainConfig
from shared.filter import GenericFilter

logger = logging.getLogger(__name__)


@dataclass
class SweepSample:
    constraints: dict[str, Any]
    pool_size: int
    n_constraints: int
    coherent: bool
    constraint_names: list[str] = field(default_factory=list)


@dataclass
class CalibrationReport:
    n_samples: int
    n_coherent: int
    n_incoherent: int
    pool_sizes: list[int]
    pool_percentiles: dict[str, float]
    zero_pool_rate: float
    small_pool_rate: float          # pool 1-3
    per_constraint_effect: dict[str, dict[str, float]]
    bracket_coverage: dict[str, dict[str, Any]]
    samples: list[SweepSample]


def _sample_one_constraint_set(config: DomainConfig) -> dict[str, Any]:
    """Sample a random constraint set from the config's registry."""
    constraints: dict[str, Any] = {}
    for cspec in config.constraints:
        if cspec.desc_constraint:
            continue
        if not cspec.sampling_values:
            continue
        include = cspec.always_include or random.random() < cspec.sampling_probability
        if include:
            constraints[cspec.name] = random.choice(cspec.sampling_values)
    return constraints


def calibration_sweep(
    catalog: pd.DataFrame,
    config: DomainConfig,
    n_samples: int = 2000,
    seed: int = 42,
) -> CalibrationReport:
    """Sample random constraint sets and record pool sizes.

    Returns a ``CalibrationReport`` with distributions, per-constraint
    marginal effects, and bracket coverage stats.
    """
    random.seed(seed)
    samples: list[SweepSample] = []

    for _ in range(n_samples):
        constraints = _sample_one_constraint_set(config)
        if not constraints:
            continue

        coherent = config.is_coherent(constraints)
        if not coherent:
            samples.append(SweepSample(
                constraints=constraints,
                pool_size=-1,
                n_constraints=len(constraints),
                coherent=False,
                constraint_names=list(constraints.keys()),
            ))
            continue

        pool = GenericFilter.apply(catalog, constraints, config.constraints)
        samples.append(SweepSample(
            constraints=constraints,
            pool_size=len(pool),
            n_constraints=len(constraints),
            coherent=True,
            constraint_names=list(constraints.keys()),
        ))

    coherent_samples = [s for s in samples if s.coherent]
    pool_sizes = [s.pool_size for s in coherent_samples]

    if pool_sizes:
        arr = np.array(pool_sizes)
        percentiles = {
            f"p{p}": float(np.percentile(arr, p))
            for p in [5, 10, 25, 50, 75, 90, 95]
        }
    else:
        percentiles = {}

    n_zero = sum(1 for p in pool_sizes if p == 0)
    n_small = sum(1 for p in pool_sizes if 1 <= p <= 3)
    n_coherent = len(coherent_samples)

    per_constraint: dict[str, dict[str, float]] = {}
    for cspec in config.constraints:
        if cspec.desc_constraint:
            continue
        with_c = [s.pool_size for s in coherent_samples if cspec.name in s.constraint_names]
        without_c = [s.pool_size for s in coherent_samples if cspec.name not in s.constraint_names]
        if with_c and without_c:
            per_constraint[cspec.name] = {
                "median_with": float(np.median(with_c)),
                "median_without": float(np.median(without_c)),
                "reduction_ratio": 1.0 - (float(np.median(with_c)) / max(float(np.median(without_c)), 1)),
                "n_with": len(with_c),
                "n_without": len(without_c),
            }

    bracket_coverage: dict[str, dict[str, Any]] = {}
    for diff_name, bracket in config.difficulty.items():
        lo, hi = bracket.pool_range
        in_range = [s for s in coherent_samples if lo <= s.pool_size <= hi]
        bracket_coverage[diff_name] = {
            "target_count": bracket.target_count,
            "viable_samples": len(in_range),
            "coverage_ratio": len(in_range) / max(n_coherent, 1),
            "sufficient": len(in_range) >= bracket.target_count * 2,
        }

    return CalibrationReport(
        n_samples=len(samples),
        n_coherent=n_coherent,
        n_incoherent=len(samples) - n_coherent,
        pool_sizes=pool_sizes,
        pool_percentiles=percentiles,
        zero_pool_rate=n_zero / max(n_coherent, 1),
        small_pool_rate=n_small / max(n_coherent, 1),
        per_constraint_effect=per_constraint,
        bracket_coverage=bracket_coverage,
        samples=samples,
    )


def auto_tune_difficulty(
    report: CalibrationReport,
    config: DomainConfig,
) -> dict[str, dict[str, Any]]:
    """Propose adjusted difficulty brackets from the sweep distribution.

    Returns a dict of ``{difficulty_name: {"pool_range": (lo, hi), ...}}``.
    Does NOT mutate the config — caller reviews and applies.
    """
    nonzero = sorted([p for p in report.pool_sizes if p > 3])
    if not nonzero:
        logger.warning("No pools > 3; constraints may be too tight")
        return {}

    arr = np.array(nonzero)
    p20 = int(np.percentile(arr, 20))
    p40 = int(np.percentile(arr, 40))
    p60 = int(np.percentile(arr, 60))
    p80 = int(np.percentile(arr, 80))

    proposed = {
        "large": {"pool_range": (4, max(p20, 5)), "min_constraints": 5},
        "medium": {"pool_range": (max(p20 - 2, 4), max(p40 + 5, p20 + 5)), "min_constraints": 4},
        "small": {"pool_range": (max(p40 - 5, p20), max(p80, p40 + 10)), "min_constraints": 3},
        "oc_feasible": {"pool_range": (1, 3), "min_constraints": 4},
        "oc_infeasible": {"pool_range": (0, 0), "min_constraints": 4},
    }

    return proposed


def print_calibration_report(report: CalibrationReport) -> None:
    """Print a human-readable calibration report."""
    print("=" * 70)
    print("CALIBRATION SWEEP REPORT")
    print("=" * 70)
    print(f"\n  Samples: {report.n_samples}")
    print(f"  Coherent: {report.n_coherent} ({report.n_coherent / max(report.n_samples, 1):.0%})")
    print(f"  Incoherent (rejected): {report.n_incoherent}")
    print(f"\n  Zero-pool rate: {report.zero_pool_rate:.1%}")
    print(f"  Small-pool rate (1-3): {report.small_pool_rate:.1%}")

    print(f"\n  Pool size percentiles:")
    for k, v in report.pool_percentiles.items():
        print(f"    {k}: {v:.0f}")

    print(f"\n  Per-constraint marginal effect:")
    for name, stats in sorted(report.per_constraint_effect.items(),
                              key=lambda x: -x[1].get("reduction_ratio", 0)):
        print(f"    {name:30s}  reduction={stats['reduction_ratio']:.0%}  "
              f"median_with={stats['median_with']:.0f}  "
              f"median_without={stats['median_without']:.0f}  "
              f"n={stats['n_with']}")

    print(f"\n  Bracket coverage:")
    for name, info in report.bracket_coverage.items():
        status = "OK" if info["sufficient"] else "INSUFFICIENT"
        print(f"    {name:20s}  viable={info['viable_samples']:4d}  "
              f"need={info['target_count'] * 2:4d}  [{status}]")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Constraint-count discovery
# ---------------------------------------------------------------------------

@dataclass
class ConstraintCountSuggestion:
    min_constraints: int
    pool_range: tuple[int, int]
    median_pool: float
    in_range_pct: float
    p25_pool: float
    p75_pool: float


@dataclass
class ConstraintCountReport:
    suggestions: dict[str, ConstraintCountSuggestion]
    sweep: dict[int, list[int]]
    catalog_size: int
    n_samples_per_count: int


def discover_constraint_counts(
    catalog: pd.DataFrame,
    config: DomainConfig,
    target_pool_sizes: dict[str, tuple[int, int]],
    n_samples_per_count: int = 300,
    max_constraint_count: int = 15,
    value_frequencies: dict[str, dict[Any, float]] | None = None,
    seed: int = 42,
) -> ConstraintCountReport:
    """Empirically discover constraint counts that produce target pool sizes.

    For each constraint count *k* (from the number of always-include
    constraints up to *max_constraint_count*), samples *n_samples_per_count*
    random constraint sets with exactly *k* constraints, picks values
    (optionally frequency-weighted), and records pool sizes via the bitmask
    pool index.

    Then for each difficulty tier in *target_pool_sizes*, picks the *k* whose
    samples most often land inside the target range.

    Returns a :class:`ConstraintCountReport` with per-tier suggestions and the
    raw sweep data.
    """
    from generation.generate_tasks import PoolIndex

    rng = random.Random(seed)
    pool_index = PoolIndex(catalog, config)

    active = [c for c in config.constraints
              if not c.desc_constraint and c.sampling_values]
    always = [c for c in active if c.always_include]
    candidates = [c for c in active if not c.always_include]
    n_always = len(always)
    max_possible = n_always + len(candidates)

    effective_max = min(max_constraint_count, max_possible)
    effective_min = max(n_always, 1)

    sweep: dict[int, list[int]] = {}

    for k in range(effective_min, effective_max + 1):
        pools: list[int] = []
        n_extra = k - n_always

        for _ in range(n_samples_per_count):
            selected = list(always)

            if n_extra > 0 and candidates:
                weights = [max(c.sampling_probability, 0.01) for c in candidates]
                avail = list(range(len(candidates)))
                for _ in range(min(n_extra, len(avail))):
                    cur_w = [weights[j] for j in avail]
                    total = sum(cur_w)
                    if total <= 0:
                        break
                    probs = [w / total for w in cur_w]
                    pick = rng.choices(range(len(avail)), weights=probs, k=1)[0]
                    selected.append(candidates[avail.pop(pick)])

            mask = pool_index.full_mask()
            for cspec in selected:
                vals = list(cspec.sampling_values)
                if not vals:
                    continue
                if value_frequencies and cspec.name in value_frequencies:
                    freq = value_frequencies[cspec.name]
                    vw = [max(freq.get(v, 0.001), 0.001) for v in vals]
                    val = rng.choices(vals, weights=vw, k=1)[0]
                else:
                    val = rng.choice(vals)
                val_mask = pool_index.get_mask(cspec.name, val)
                mask = mask & val_mask

            pools.append(pool_index.pool_size(mask))

        sweep[k] = pools

    suggestions: dict[str, ConstraintCountSuggestion] = {}

    for tier, (lo, hi) in target_pool_sizes.items():
        mid = (lo + hi) / 2
        best_k: int | None = None
        best_score = -1.0

        for k in sorted(sweep.keys()):
            pools = sweep[k]
            median = float(np.median(pools))
            in_range = sum(1 for p in pools if lo <= p <= hi)
            in_range_frac = in_range / len(pools)
            distance = abs(median - mid) / max(mid, 1) if mid > 0 else abs(median)
            score = in_range_frac - distance * 0.01
            if score > best_score:
                best_score = score
                best_k = k

        if best_k is not None:
            best_pools = sweep[best_k]
            suggestions[tier] = ConstraintCountSuggestion(
                min_constraints=best_k,
                pool_range=(lo, hi),
                median_pool=float(np.median(best_pools)),
                in_range_pct=sum(1 for p in best_pools if lo <= p <= hi)
                / len(best_pools) * 100,
                p25_pool=float(np.percentile(best_pools, 25)),
                p75_pool=float(np.percentile(best_pools, 75)),
            )

    return ConstraintCountReport(
        suggestions=suggestions,
        sweep=sweep,
        catalog_size=len(catalog),
        n_samples_per_count=n_samples_per_count,
    )


def print_constraint_count_report(report: ConstraintCountReport) -> None:
    """Print a human-readable constraint-count discovery report."""
    print("=" * 70)
    print("  CONSTRAINT COUNT DISCOVERY")
    print(f"  Catalog: {report.catalog_size:,} rows | "
          f"{report.n_samples_per_count} samples per constraint count")
    print("=" * 70)

    print(f"\n  {'k':>4s}  {'median':>8s}  {'p25':>8s}  {'p75':>8s}  {'zero%':>6s}")
    print(f"  {'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 6}")
    for k in sorted(report.sweep.keys()):
        pools = report.sweep[k]
        arr = np.array(pools)
        zero_pct = sum(1 for p in pools if p == 0) / len(pools) * 100
        print(f"  {k:4d}  {np.median(arr):8.0f}  {np.percentile(arr, 25):8.0f}  "
              f"{np.percentile(arr, 75):8.0f}  {zero_pct:5.1f}%")

    print(f"\n  Suggestions per difficulty tier:")
    print(f"  {'tier':>20s}  {'min_c':>5s}  {'pool_range':>14s}  "
          f"{'median':>7s}  {'in_range':>8s}  {'IQR':>16s}")
    print(f"  {'─' * 20}  {'─' * 5}  {'─' * 14}  {'─' * 7}  {'─' * 8}  {'─' * 16}")
    for tier, s in report.suggestions.items():
        print(f"  {tier:>20s}  {s.min_constraints:5d}  "
              f"{'(' + str(s.pool_range[0]) + ', ' + str(s.pool_range[1]) + ')':>14s}  "
              f"{s.median_pool:7.0f}  {s.in_range_pct:7.0f}%  "
              f"[{s.p25_pool:.0f}, {s.p75_pool:.0f}]")

    print(f"\n  ── Copy-paste snippet ──")
    print("  DIFFICULTY_OVERRIDES = {")
    for tier, s in report.suggestions.items():
        print(f'      "{tier}": {{"pool_range": {s.pool_range}, '
              f'"min_constraints": {s.min_constraints}}},')
    print("  }")
    print("=" * 70)
