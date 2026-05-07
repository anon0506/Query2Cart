"""Catalog hygiene: clean rows, prune sampling values, compute value frequencies.

Run ``clean_catalog`` after config generation and before calibration sweeps to
remove phantom rows (insufficient attribute coverage) and ultra-rare or garbage
sampling values that create near-certain zero pools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.config import AttrType, DomainConfig
from shared.filter import OPERATORS

logger = logging.getLogger(__name__)

_GARBAGE_TAG_VALUES_HARDCODED = frozenset({
    "en:undefined", "undefined", "not-applicable", "unknown",
    "none", "n/a", "na", "null", "nan", "", "-", "?",
})


# ---------------------------------------------------------------------------
# LLM-backed garbage value detection
# ---------------------------------------------------------------------------

def identify_garbage_values_llm(
    config: DomainConfig,
    catalog: pd.DataFrame,
    *,
    model: str = "gpt-4.1-mini",
    max_values_per_column: int = 40,
) -> frozenset[str]:
    """Use an LLM to identify garbage/sentinel values across all constraint columns.

    Augments the hardcoded ``_GARBAGE_TAG_VALUES_HARDCODED`` with domain-aware
    detection.  The LLM sees each column's value distribution and flags entries
    that are meaningless for user-facing filtering.

    Returns the union of hardcoded and LLM-identified garbage values (lowercased).
    """
    from shared.llm import call_llm_json, format_prompt_template, resolve_prompt_path

    template = Path(resolve_prompt_path("identify_garbage_values.txt")).read_text(
        encoding="utf-8"
    )

    domain_desc = (config.prompt_fragments.domain_description or "").strip() or config.name
    all_garbage: set[str] = set(_GARBAGE_TAG_VALUES_HARDCODED)

    columns_checked = 0
    for cspec in config.constraints:
        if cspec.desc_constraint or not cspec.sampling_values:
            continue
        if cspec.attribute not in catalog.columns:
            continue

        col = catalog[cspec.attribute]
        col_values = cspec.sampling_values

        value_rows: list[str] = []
        for val in col_values[:max_values_per_column]:
            val_str = str(val)
            if cspec.attribute in catalog.columns:
                from shared.filter import OPERATORS as _ops
                op_fn = _ops.get(cspec.operator)
                if op_fn is not None:
                    hit_count = int(op_fn(col, val).sum())
                else:
                    hit_count = 0
            else:
                hit_count = 0
            pct = hit_count / max(len(catalog), 1) * 100
            value_rows.append(f"  {val_str:50s}  count={hit_count:6d}  ({pct:.2f}%)")

        if not value_rows:
            continue

        attr_spec = config._attribute_by_name.get(cspec.attribute)
        col_desc = (attr_spec.display_name if attr_spec else cspec.attribute).replace("_", " ")
        col_type = (attr_spec.attr_type.value if attr_spec else "unknown")

        prompt = format_prompt_template(
            template,
            domain_description=domain_desc,
            column_name=cspec.attribute,
            column_description=col_desc,
            column_type=col_type,
            total_rows=str(len(catalog)),
            value_table="\n".join(value_rows),
            item_noun_plural=config.item_noun_plural,
        )

        try:
            result = call_llm_json(
                prompt, model,
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            garbage_vals = result.get("garbage_values", [])
            for gv in garbage_vals:
                val_lower = str(gv).lower().strip()
                if val_lower:
                    all_garbage.add(val_lower)
                    logger.info(
                        "LLM flagged garbage value '%s' in %s: %s",
                        gv, cspec.attribute,
                        result.get("reasoning", {}).get(str(gv), ""),
                    )
            columns_checked += 1
        except Exception:
            logger.warning(
                "LLM garbage detection failed for column '%s'; using hardcoded list only",
                cspec.attribute, exc_info=True,
            )

    logger.info(
        "LLM garbage detection: checked %d columns, total garbage values: %d (hardcoded=%d, llm-added=%d)",
        columns_checked, len(all_garbage),
        len(_GARBAGE_TAG_VALUES_HARDCODED),
        len(all_garbage) - len(_GARBAGE_TAG_VALUES_HARDCODED),
    )
    return frozenset(all_garbage)


# ---------------------------------------------------------------------------
# Row-level coverage
# ---------------------------------------------------------------------------

def clean_catalog(
    catalog: pd.DataFrame,
    config: DomainConfig,
    min_row_coverage: float = 0.6,
) -> pd.DataFrame:
    """Drop rows where too many filterable columns are null or empty.

    Also treats empty lists in set-valued columns as missing.
    Returns a copy of the cleaned catalog.
    """
    filterable_cols = [
        a.name for a in config.attributes
        if a.filterable and a.name in catalog.columns
    ]
    if not filterable_cols:
        logger.warning("No filterable columns found in catalog; skipping row pruning")
        return catalog.copy()

    presence = pd.DataFrame(index=catalog.index)
    for col_name in filterable_cols:
        attr = next((a for a in config.attributes if a.name == col_name), None)
        col = catalog[col_name]

        if attr and attr.attr_type == AttrType.SET_VALUED:
            presence[col_name] = col.apply(_has_meaningful_tags)
        else:
            presence[col_name] = col.notna()

    row_coverage = presence.mean(axis=1)
    keep_mask = row_coverage >= min_row_coverage

    n_dropped = (~keep_mask).sum()
    logger.info(
        "Row coverage filter: keeping %d/%d rows (dropped %d with coverage < %.0f%%)",
        keep_mask.sum(), len(catalog), n_dropped, min_row_coverage * 100,
    )
    return catalog[keep_mask].copy()


def _has_meaningful_tags(cell: Any, garbage: frozenset[str] | None = None) -> bool:
    """True if cell contains at least one non-garbage tag."""
    if cell is None:
        return False
    if isinstance(cell, float) and pd.isna(cell):
        return False
    garbage_set = garbage or _GARBAGE_TAG_VALUES_HARDCODED
    if isinstance(cell, (list, tuple, np.ndarray)):
        tags = [str(t).lower().strip() for t in cell]
        return any(t not in garbage_set for t in tags)
    return False


# ---------------------------------------------------------------------------
# Value frequency computation
# ---------------------------------------------------------------------------

def compute_value_frequencies(
    catalog: pd.DataFrame,
    config: DomainConfig,
) -> dict[str, dict[Any, float]]:
    """For each constraint's sampling values, compute the single-constraint hit rate.

    Returns ``{constraint_name: {value: hit_rate}}`` where hit_rate is the
    fraction of catalog rows matching that one constraint with that value.
    """
    n = len(catalog)
    if n == 0:
        return {}

    freqs: dict[str, dict[Any, float]] = {}
    for cspec in config.constraints:
        if cspec.desc_constraint or not cspec.sampling_values:
            continue
        if cspec.attribute not in catalog.columns:
            continue

        op_fn = OPERATORS.get(cspec.operator)
        if op_fn is None:
            continue

        col = catalog[cspec.attribute]
        value_rates: dict[Any, float] = {}
        for val in cspec.sampling_values:
            mask = op_fn(col, val)
            value_rates[val] = float(mask.sum()) / n
        freqs[cspec.name] = value_rates

    return freqs


# ---------------------------------------------------------------------------
# Sampling value pruning
# ---------------------------------------------------------------------------

def prune_sampling_values(
    config: DomainConfig,
    catalog: pd.DataFrame,
    min_hit_rate: float = 0.005,
    garbage_values: frozenset[str] | None = None,
) -> dict[str, dict[Any, float]]:
    """Remove garbage and ultra-rare values from each constraint's sampling_values.

    Mutates ``config.constraints`` in place (same pattern as existing
    ``_refine_sampling_probabilities``).  Returns the computed value frequencies
    for downstream use (e.g. frequency-weighted sampling).

    When ``garbage_values`` is provided (e.g. from :func:`identify_garbage_values_llm`),
    those are used directly.  Otherwise falls back to the hardcoded set.
    """
    garbage = garbage_values or _GARBAGE_TAG_VALUES_HARDCODED
    freqs = compute_value_frequencies(catalog, config)

    for cspec in config.constraints:
        if cspec.desc_constraint or not cspec.sampling_values:
            continue

        rates = freqs.get(cspec.name, {})
        original = list(cspec.sampling_values)
        cleaned: list[Any] = []

        for val in original:
            val_str = str(val).lower().strip()
            if val_str in garbage:
                logger.info("Pruned garbage value '%s' from %s", val, cspec.name)
                continue

            rate = rates.get(val, 0.0)
            if rate < min_hit_rate:
                logger.info(
                    "Pruned rare value '%s' from %s (hit_rate=%.4f < %.4f)",
                    val, cspec.name, rate, min_hit_rate,
                )
                continue

            cleaned.append(val)

        if len(cleaned) < len(original):
            logger.info(
                "Constraint '%s': %d → %d sampling values after pruning",
                cspec.name, len(original), len(cleaned),
            )
        cspec.sampling_values = cleaned

        # Update frequencies to reflect only retained values
        if cspec.name in freqs:
            freqs[cspec.name] = {v: r for v, r in freqs[cspec.name].items() if v in cleaned}

    return freqs


# ---------------------------------------------------------------------------
# Constraint viability audit
# ---------------------------------------------------------------------------

def audit_constraint_viability(config: DomainConfig) -> list[str]:
    """Log warnings for constraints with too few viable sampling values.

    Returns list of constraint names that are non-viable (< 2 values).
    """
    non_viable: list[str] = []
    for cspec in config.constraints:
        if cspec.desc_constraint:
            continue
        n_vals = len(cspec.sampling_values or [])
        if n_vals == 0:
            logger.warning(
                "Constraint '%s' has NO sampling values — setting probability to 0",
                cspec.name,
            )
            cspec.sampling_probability = 0.0
            non_viable.append(cspec.name)
        elif n_vals < 2:
            logger.warning(
                "Constraint '%s' has only %d sampling value — consider removing or demoting",
                cspec.name, n_vals,
            )
            non_viable.append(cspec.name)
    return non_viable


# ---------------------------------------------------------------------------
# All-in-one entry point
# ---------------------------------------------------------------------------

def clean_catalog_and_config(
    catalog: pd.DataFrame,
    config: DomainConfig,
    min_row_coverage: float = 0.6,
    min_value_hit_rate: float = 0.005,
    garbage_model: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, float]]]:
    """Run all hygiene steps: row pruning → LLM garbage detection → value pruning → viability audit.

    When ``garbage_model`` is provided, an LLM identifies domain-specific garbage
    values in addition to the hardcoded sentinel list.  Pass ``None`` (default)
    to use only the hardcoded list.

    Returns ``(cleaned_catalog, value_frequencies)``.  Mutates ``config``
    in place for sampling value changes.
    """
    cleaned = clean_catalog(catalog, config, min_row_coverage)

    garbage_vals: frozenset[str] | None = None
    if garbage_model:
        garbage_vals = identify_garbage_values_llm(
            config, cleaned, model=garbage_model,
        )

    freqs = prune_sampling_values(config, cleaned, min_value_hit_rate, garbage_values=garbage_vals)

    non_viable = audit_constraint_viability(config)
    if non_viable:
        logger.warning("Non-viable constraints after pruning: %s", non_viable)

    logger.info(
        "Catalog hygiene complete: %d → %d rows, %d constraints viable",
        len(catalog), len(cleaned),
        sum(1 for c in config.constraints
            if not c.desc_constraint and (c.sampling_values and len(c.sampling_values) >= 2)),
    )

    return cleaned, freqs


def tool_facing_catalog_columns(
    config: DomainConfig,
    catalog: pd.DataFrame,
) -> list[str]:
    """Ordered columns for a config-aligned (tool-facing) catalog slice.

    Includes ``id_column``, every :class:`AttributeSpec` name present in
    ``catalog``, then any constraint ``attribute`` or preference catalog field
    not already listed.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def add(col: str) -> None:
        if col in catalog.columns and col not in seen:
            ordered.append(col)
            seen.add(col)

    add(config.id_column)
    for a in config.attributes:
        add(a.name)
    for c in config.constraints:
        add(c.attribute)
    for p in config.preference_attributes:
        add(p.get_catalog_field())
    return ordered


def project_catalog_for_tools(
    catalog: pd.DataFrame,
    config: DomainConfig,
) -> pd.DataFrame:
    """Restrict ``catalog`` to columns declared in the domain config."""
    cols = tool_facing_catalog_columns(config, catalog)
    if not cols:
        raise ValueError(
            "No tool-facing columns intersect the catalog; check id_column and config attributes."
        )
    return catalog[cols].copy()
