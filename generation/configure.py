"""Generate a draft DomainConfig from a triage result and catalog profile.

The output is a config that is *functionally complete* but needs human review:
constraint sampling values need rounding, coherence rules need expert validation,
and trigger keywords need domain-specific tuning.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.config import (
    AttrType,
    AttributeSpec,
    ConstraintOp,
    ConstraintSpec,
    DifficultyBracket,
    DomainConfig,
    PreferenceAttributeSpec,
    PromptFragments,
    TriggerSpec,
    ViolationTrigger,
)
from shared.filter import GenericFilter
from shared.llm import call_llm_json, format_prompt_template, resolve_prompt_path
from generation.triage import ColumnAssignment, ColumnRole, TriageResult

logger = logging.getLogger(__name__)

_GARBAGE_STRINGS = frozenset({
    "", "null", "nan", "n/a", "unknown", "none", "na", "nil", "-", "?",
    "not specified", "not available", "undefined",
})

# ---------------------------------------------------------------------------
# Attribute generation
# ---------------------------------------------------------------------------

_KIND_TO_ATTR = {
    "int_numeric": AttrType.NUMERIC,
    "float_numeric": AttrType.NUMERIC,
    "datetime": AttrType.NUMERIC,
    "string_categorical": AttrType.CATEGORICAL,
    "list_of_values": AttrType.SET_VALUED,
    "bool": AttrType.BOOLEAN,
    "string_long_text": AttrType.TEXT,
    "complex_nested": AttrType.TEXT,
}


def _make_attribute(col: ColumnAssignment) -> AttributeSpec:
    attr_type = _KIND_TO_ATTR.get(col.inferred_kind, AttrType.TEXT)
    return AttributeSpec(
        name=col.column,
        display_name=col.user_facing_name or col.column.replace("_", " ").title(),
        attr_type=attr_type,
        filterable=col.role in (ColumnRole.HARD_FILTER, ColumnRole.SET_FILTER),
        required=col.null_rate < 0.02 and col.role in (ColumnRole.HARD_FILTER, ColumnRole.ID),
        coverage_threshold=max(0.0, round(1.0 - col.null_rate - 0.1, 2)),
        embedding_field=col.role == ColumnRole.EMBEDDING_TEXT,
        preference_eligible=col.role == ColumnRole.SOFT_PREFERENCE,
        preference_direction=col.preference_direction or None,
        popularity_proxy=False,
    )


# ---------------------------------------------------------------------------
# Value sanitization
# ---------------------------------------------------------------------------

def _sanitize_sampling_values(values: list[Any], kind: str) -> list[Any]:
    """Remove garbage values from sampling lists.

    Handles empty strings, null-like sentinels, NaN/inf for numerics,
    and unwraps serialized JSON objects (e.g. {"lang": "fr", "text": "Gel"}).
    """
    cleaned: list[Any] = []
    for v in values:
        if v is None:
            continue

        if kind in ("int_numeric", "float_numeric"):
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                cleaned.append(v)
            except (ValueError, TypeError):
                continue
            continue

        if kind == "datetime":
            try:
                ts = pd.Timestamp(v)
                if pd.isna(ts):
                    continue
                cleaned.append(ts.strftime("%Y-%m-%d"))
            except (ValueError, TypeError, OSError):
                continue
            continue

        s = str(v).strip()
        if s.lower() in _GARBAGE_STRINGS:
            continue

        if s.startswith("{"):
            try:
                obj = json.loads(s)
                text = obj.get("text") or obj.get("value") or obj.get("name")
                if text and str(text).strip():
                    cleaned.append(str(text).strip())
                    continue
            except (json.JSONDecodeError, AttributeError):
                pass

        cleaned.append(v)

    return cleaned


# ---------------------------------------------------------------------------
# Adaptive sampling
# ---------------------------------------------------------------------------

def _derive_sampling_values(
    col_summary: dict[str, Any],
    direction: str,
    n_rows: int = 0,
) -> list[Any]:
    """Derive constraint sampling values using coverage-based selection.

    For categorical/set-valued columns, walks the frequency list from most
    common to least common, keeping values until their cumulative share
    reaches ``target_coverage`` (default 90%).  This naturally adapts to any
    distribution shape (Zipf, uniform, long-tail) without rigid cardinality
    tiers.

    For numeric columns, uses percentile-based values rounded to
    human-friendly numbers.
    """
    details = col_summary.get("details", {})
    kind = col_summary.get("inferred_kind", "")

    if kind == "datetime" and "min" in details and "max" in details:
        lo, hi = details["min"], details["max"]
        if lo == hi:
            pcts = [lo]
        else:
            pcts = [details.get(f"p{p}") for p in [5, 25, 50, 75, 95]]
            pcts = [p for p in pcts if p is not None]
            if not pcts:
                n_steps = 8
                step = (hi - lo) / n_steps
                pcts = [lo + step * i for i in range(1, n_steps)]
        rounded = _round_human(pcts, precision=0)
        iso: list[str] = []
        for p in rounded:
            try:
                iso.append(pd.Timestamp(float(p), unit="s").strftime("%Y-%m-%d"))
            except (ValueError, OSError, OverflowError):
                continue
        return _sanitize_sampling_values(iso, kind)

    if kind in ("string_categorical", "list_of_values"):
        full_freq = details.get("value_frequencies_full") or details.get("element_frequencies_full", [])
        if not full_freq:
            full_freq = details.get("top_10_value_frequencies") or details.get("top_10_element_frequencies", [])

        if full_freq:
            values = _coverage_based_sampling(full_freq)
            return _sanitize_sampling_values(values, kind)

    if "min" in details and "max" in details:
        lo, hi = details["min"], details["max"]
        if lo == hi:
            return _sanitize_sampling_values([lo], kind)
        pcts = [details.get(f"p{p}") for p in [5, 25, 50, 75, 95]]
        pcts = [p for p in pcts if p is not None]
        if not pcts:
            n_steps = 8
            step = (hi - lo) / n_steps
            pcts = [lo + step * i for i in range(1, n_steps)]

        is_int = kind == "int_numeric"
        return _round_human(pcts, precision=0 if is_int else 1)

    if kind == "bool":
        return [True]

    return []


_COVERAGE_TARGET = 0.90
_COVERAGE_MIN_VALUES = 3
_COVERAGE_TAIL_DROP = 0.002
_COVERAGE_MAX_VALUES = 50


def _coverage_based_sampling(
    freq_list: list[dict[str, Any]],
    *,
    target_coverage: float = _COVERAGE_TARGET,
    min_values: int = _COVERAGE_MIN_VALUES,
    tail_drop: float = _COVERAGE_TAIL_DROP,
    max_values: int = _COVERAGE_MAX_VALUES,
) -> list[Any]:
    """Select sampling values by walking the frequency list until cumulative
    share reaches *target_coverage*.

    Adapts to any distribution shape without cliff effects:
    - Zipf-like (few values dominate): keeps only the head values
    - Uniform (many similar frequencies): keeps most values
    - Long-tail: keeps the meaningful head, drops ultra-rare tail
    """
    sorted_freq = sorted(freq_list, key=lambda x: x.get("share", 0), reverse=True)

    values: list[Any] = []
    cumulative = 0.0

    for item in sorted_freq:
        if len(values) >= max_values:
            break

        share = item.get("share", 0)

        if len(values) >= min_values:
            if cumulative >= target_coverage:
                break
            if share < tail_drop:
                break

        values.append(item["value"])
        cumulative += share

    return values


def _round_human(values: list[float], precision: int = 0) -> list[Any]:
    """Round values to human-friendly numbers."""
    if precision == 0:
        return sorted(set(int(round(v)) for v in values))
    return sorted(set(round(v, precision) for v in values))


# ---------------------------------------------------------------------------
# Type-role guard
# ---------------------------------------------------------------------------

_HARD_FILTER_SUPPORTED_KINDS = frozenset({
    "int_numeric", "float_numeric", "string_categorical", "bool", "datetime",
})
_SET_FILTER_SUPPORTED_KINDS = frozenset({"list_of_values"})
_SOFT_PREFERENCE_SUPPORTED_KINDS = frozenset({
    "int_numeric", "float_numeric", "string_categorical",
})


def _apply_type_role_guard(col: ColumnAssignment) -> ColumnAssignment:
    """Downgrade columns whose inferred_kind cannot support their assigned role.

    Prevents downstream states where an attribute is marked filterable but has
    zero constraint specs (confuses calibration and task generation).
    """
    kind = col.inferred_kind

    if col.role == ColumnRole.HARD_FILTER and kind not in _HARD_FILTER_SUPPORTED_KINDS:
        logger.warning(
            "Type-role guard: column '%s' (kind=%s) cannot be hard_filter; "
            "downgrading to drop",
            col.column, kind,
        )
        col.role = ColumnRole.DROP
        col.note = (col.note + "; " if col.note else "") + f"downgraded from hard_filter (kind={kind})"

    elif col.role == ColumnRole.SET_FILTER and kind not in _SET_FILTER_SUPPORTED_KINDS:
        logger.warning(
            "Type-role guard: column '%s' (kind=%s) cannot be set_filter; "
            "downgrading to drop",
            col.column, kind,
        )
        col.role = ColumnRole.DROP
        col.note = (col.note + "; " if col.note else "") + f"downgraded from set_filter (kind={kind})"

    elif col.role == ColumnRole.SOFT_PREFERENCE and kind not in _SOFT_PREFERENCE_SUPPORTED_KINDS:
        logger.warning(
            "Type-role guard: column '%s' (kind=%s) cannot be soft_preference; "
            "downgrading to drop",
            col.column, kind,
        )
        col.role = ColumnRole.DROP
        col.note = (col.note + "; " if col.note else "") + f"downgraded from soft_preference (kind={kind})"

    return col


# ---------------------------------------------------------------------------
# Constraint generation (polarity-aware)
# ---------------------------------------------------------------------------

def _make_constraint(
    col: ColumnAssignment,
    col_summary: dict[str, Any],
    n_rows: int = 0,
) -> list[ConstraintSpec]:
    """Generate constraint specs from a column assignment and its profile stats."""
    specs: list[ConstraintSpec] = []
    kind = col.inferred_kind

    if col.role == ColumnRole.HARD_FILTER:
        if kind in ("int_numeric", "float_numeric"):
            direction = col.constraint_direction or "max"
            op = ConstraintOp.LTE if direction == "max" else ConstraintOp.GTE
            label_prefix = "Maximum" if direction == "max" else "Minimum"
            unit = ""
            name_suffix = "_max" if direction == "max" else "_min"
            name = f"{col.column}{name_suffix}"

            sampling = _derive_sampling_values(col_summary, direction, n_rows)
            soft_dir = "minimize" if direction == "max" else "maximize"
            specs.append(ConstraintSpec(
                name=name,
                attribute=col.column,
                operator=op,
                display_template=f"{label_prefix} {col.user_facing_name or col.column}: {{value}}{unit}",
                value_type="int" if kind == "int_numeric" else "float",
                sampling_values=sampling,
                sampling_probability=0.5,
                demotable=True,
                soft_direction=soft_dir,
            ))

        elif kind == "string_categorical":
            sampling = _derive_sampling_values(col_summary, "eq", n_rows)
            n_vals = len(sampling) if sampling else col.n_unique
            op = ConstraintOp.EQ if n_vals <= 10 else ConstraintOp.IN_SET
            display_name = col.user_facing_name or col.column
            specs.append(ConstraintSpec(
                name=col.column,
                attribute=col.column,
                operator=op,
                display_template=f"{display_name}: {{value}}",
                value_type="str" if op == ConstraintOp.EQ else "list[str]",
                sampling_values=sampling,
                sampling_probability=0.35,
                demotable=True,
                soft_direction="match",
            ))

        elif kind == "datetime":
            direction = col.constraint_direction or "max"
            op = ConstraintOp.LTE if direction == "max" else ConstraintOp.GTE
            label_prefix = "On or before" if direction == "max" else "On or after"
            name_suffix = "_max" if direction == "max" else "_min"
            name = f"{col.column}{name_suffix}"

            sampling = _derive_sampling_values(col_summary, direction, n_rows)
            soft_dir = "minimize" if direction == "max" else "maximize"
            specs.append(ConstraintSpec(
                name=name,
                attribute=col.column,
                operator=op,
                display_template=(
                    f"{label_prefix} {col.user_facing_name or col.column}: {{value}}"
                ),
                value_type="str",
                sampling_values=sampling,
                sampling_probability=0.5,
                demotable=True,
                soft_direction=soft_dir,
            ))

        elif kind == "bool":
            specs.append(ConstraintSpec(
                name=col.column,
                attribute=col.column,
                operator=ConstraintOp.BOOLEAN,
                display_template=f"{col.user_facing_name or col.column}: {{value}}",
                value_type="bool",
                sampling_values=[True],
                sampling_probability=0.25,
                demotable=True,
                soft_direction="match",
            ))

    elif col.role == ColumnRole.SET_FILTER:
        sampling = _derive_sampling_values(col_summary, "contains", n_rows)
        membership = col.membership_type or "contains_any"
        polarity = getattr(col, "constraint_polarity", "") or "positive"

        if polarity == "negative":
            op = (ConstraintOp.NOT_CONTAINS if membership == "contains"
                  else ConstraintOp.NOT_CONTAINS_ALL if membership == "contains_all"
                  else ConstraintOp.NOT_CONTAINS_ANY)
            name_suffix = "_excludes"
            display_verb = "must NOT contain"
        else:
            op = (ConstraintOp.CONTAINS if membership == "contains"
                  else ConstraintOp.CONTAINS_ALL if membership == "contains_all"
                  else ConstraintOp.CONTAINS_ANY)
            name_suffix = "_includes"
            display_verb = "must include"

        display_name = col.user_facing_name or col.column
        is_positive = polarity != "negative"
        specs.append(ConstraintSpec(
            name=f"{col.column}{name_suffix}",
            attribute=col.column,
            operator=op,
            display_template=f"{display_name} {display_verb}: {{value}}",
            value_type="str",
            sampling_values=sampling,
            sampling_probability=0.40,
            demotable=is_positive,
            soft_direction="match" if is_positive else None,
        ))

    return specs


# ---------------------------------------------------------------------------
# Violation trigger descriptions (deterministic)
# ---------------------------------------------------------------------------

_VIOLATION_TEMPLATES: dict[ConstraintOp, str] = {
    ConstraintOp.LTE: "Recommended {item} exceeds the user's {display} limit",
    ConstraintOp.GTE: "Recommended {item} falls below the user's {display} requirement",
    ConstraintOp.EQ: "Recommended {item} does not match the required {display}",
    ConstraintOp.NEQ: "Recommended {item} matches the excluded {display}",
    ConstraintOp.IN_SET: "Recommended {item} is not in the required {display} set",
    ConstraintOp.CONTAINS: "Recommended {item} does not contain required {display}",
    ConstraintOp.CONTAINS_ANY: "Recommended {item} does not include any of the required {display}",
    ConstraintOp.CONTAINS_ALL: "Recommended {item} is missing required {display}",
    ConstraintOp.NOT_CONTAINS: "Recommended {item} contains unwanted {display}",
    ConstraintOp.NOT_CONTAINS_ANY: "Recommended {item} contains an avoided {display}",
    ConstraintOp.NOT_CONTAINS_ALL: "Recommended {item} contains all of the avoided {display}",
    ConstraintOp.BOOLEAN: "Recommended {item} does not match the {display} requirement",
    ConstraintOp.SUBSTRING: "Recommended {item} does not match {display}",
    ConstraintOp.RANGE: "Recommended {item} is outside the {display} range",
}


def _make_violation_description(spec: ConstraintSpec, item_noun: str) -> str:
    display_name = spec.display_template.split(":")[0].strip().split("{")[0].strip()
    if not display_name:
        display_name = spec.name
    template = _VIOLATION_TEMPLATES.get(spec.operator, "Recommended {item} violates {display}")
    return template.format(item=item_noun, display=display_name.lower())


def _make_violation_trigger_name(spec: ConstraintSpec) -> str:
    """Generate a violation trigger name from a constraint spec."""
    op = spec.operator
    attr = spec.attribute.replace("_tags", "").replace("_", "_")

    if op in (ConstraintOp.NOT_CONTAINS, ConstraintOp.NOT_CONTAINS_ANY, ConstraintOp.NOT_CONTAINS_ALL):
        return f"shown_{attr}_present"
    if op == ConstraintOp.LTE:
        return f"shown_high_{attr}"
    if op == ConstraintOp.GTE:
        return f"shown_low_{attr}"
    return f"shown_wrong_{attr}"


# ---------------------------------------------------------------------------
# Trigger generation (LLM-assisted)
# ---------------------------------------------------------------------------

def _generate_triggers(
    constraints: list[ConstraintSpec],
    item_noun: str,
    *,
    domain_description: str,
    preference_attributes: list[PreferenceAttributeSpec],
    model: str = "gpt-4.1-mini",
) -> tuple[list[TriggerSpec], list[ViolationTrigger]]:
    pref_attr_names = {p.attribute for p in preference_attributes}

    constraint_summaries = "\n".join(
        f"- {c.name}: {c.display_template} (operator: {c.operator.value})"
        for c in constraints if not c.desc_constraint
    )
    if not (domain_description or "").strip():
        raise ValueError("domain_description is required for trigger generation")

    pref_summary = "\n".join(
        f"- {p.attribute}: direction={', '.join(p.directions)}"
        for p in preference_attributes
    )

    template = Path(resolve_prompt_path("suggest_triggers.txt")).read_text(
        encoding="utf-8"
    )
    user_prompt = format_prompt_template(
        template,
        domain_description=domain_description,
        item_noun=item_noun,
        constraint_summaries=constraint_summaries,
        preference_summaries=pref_summary,
    )
    try:
        data = call_llm_json(
            user_prompt,
            model,
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning("LLM trigger generation failed; producing minimal defaults", exc_info=True)
        data = {"triggers": [], "violation_triggers": []}

    triggers = []
    for t in data.get("triggers", []):
        raw_prefs = t.get("unlocks_preferences", [])
        validated_prefs = [p for p in raw_prefs if p in pref_attr_names]
        if raw_prefs and not validated_prefs:
            logger.debug(
                "Trigger '%s' unlocks_preferences %s matched no actual preference attributes %s",
                t.get("name"), raw_prefs, pref_attr_names,
            )
        triggers.append(TriggerSpec(
            name=t.get("name", ""),
            keywords=t.get("keywords", []),
            description=t.get("description", ""),
            unlocks_constraints=t.get("unlocks_constraints", []),
            unlocks_preferences=validated_prefs,
        ))

    violation_triggers = []
    constraint_names_with_trigger = set()
    for vt in data.get("violation_triggers", []):
        cn = vt.get("constraint_name", "")
        constraint_names_with_trigger.add(cn)
        spec = next((c for c in constraints if c.name == cn), None)
        if spec:
            desc = _make_violation_description(spec, item_noun)
        else:
            desc = vt.get("description", "")
        violation_triggers.append(ViolationTrigger(
            name=vt.get("name", ""),
            description=desc,
            constraint_name=cn,
        ))

    for c in constraints:
        if c.desc_constraint:
            continue
        if c.name not in constraint_names_with_trigger:
            vt_name = _make_violation_trigger_name(c)
            violation_triggers.append(ViolationTrigger(
                name=vt_name,
                description=_make_violation_description(c, item_noun),
                constraint_name=c.name,
            ))

    return triggers, violation_triggers


# ---------------------------------------------------------------------------
# Preference attribute generation (with random hard_filter promotion)
# ---------------------------------------------------------------------------

_HARD_FILTER_PROMOTION_DIRECTIONS: dict[str, str] = {
    "max": "minimize",
    "min": "maximize",
}

_MIN_PREFERENCE_ATTRIBUTES = 3


def _generate_preference_attributes(
    triage: TriageResult,
) -> list[PreferenceAttributeSpec]:
    """Generate preference attributes from soft_preference columns.

    If fewer than _MIN_PREFERENCE_ATTRIBUTES result, randomly promotes
    eligible hard_filter numeric columns as additional preferences.
    """
    pref_cols = triage.by_role(ColumnRole.SOFT_PREFERENCE)
    specs = []
    used_attrs: set[str] = set()
    for col in pref_cols:
        direction = col.preference_direction or "maximize"
        specs.append(PreferenceAttributeSpec(
            attribute=col.column,
            directions=[direction],
            catalog_field=col.column,
        ))
        used_attrs.add(col.column)

    if len(specs) >= _MIN_PREFERENCE_ATTRIBUTES:
        return specs

    promotion_candidates = []
    for col in triage.columns:
        if col.column in used_attrs:
            continue
        if col.role != ColumnRole.HARD_FILTER:
            continue
        if col.inferred_kind not in ("int_numeric", "float_numeric", "datetime"):
            continue
        direction = _HARD_FILTER_PROMOTION_DIRECTIONS.get(
            col.constraint_direction or "max", "minimize",
        )
        promotion_candidates.append((col, direction))

    for col in triage.columns:
        if col.column in used_attrs:
            continue
        if col.role != ColumnRole.HARD_FILTER:
            continue
        if col.inferred_kind != "string_categorical":
            continue
        promotion_candidates.append((col, "match"))

    random.shuffle(promotion_candidates)
    needed = _MIN_PREFERENCE_ATTRIBUTES - len(specs)
    for col, direction in promotion_candidates[:needed]:
        specs.append(PreferenceAttributeSpec(
            attribute=col.column,
            directions=[direction],
            catalog_field=col.column,
        ))
        used_attrs.add(col.column)
        logger.info(
            "Promoted hard_filter '%s' to additional preference (direction=%s) "
            "to reach minimum of %d preferences",
            col.column, direction, _MIN_PREFERENCE_ATTRIBUTES,
        )

    if len(specs) < _MIN_PREFERENCE_ATTRIBUTES:
        logger.warning(
            "Only %d preference attributes found (target: %d). "
            "Domain may need manual preference additions.",
            len(specs), _MIN_PREFERENCE_ATTRIBUTES,
        )

    return specs


# ---------------------------------------------------------------------------
# LLM-driven sampling probabilities
# ---------------------------------------------------------------------------

def _get_realism_weights(
    constraints: list[ConstraintSpec],
    item_noun: str,
    domain_description: str,
    model: str = "gpt-4.1-mini",
) -> dict[str, tuple[int, bool]]:
    """Stage A: ask the LLM for realism weights (1-5 scale) per constraint.

    Returns ``{constraint_name: (weight, always_include)}``.
    """
    constraint_list = "\n".join(
        f"- {c.name}: {c.display_template} (operator: {c.operator.value}, "
        f"n_sampling_values: {len(c.sampling_values or [])})"
        for c in constraints
        if not c.desc_constraint and c.sampling_values
    )

    template = Path(resolve_prompt_path("realism_weights.txt")).read_text(
        encoding="utf-8"
    )
    prompt = format_prompt_template(
        template,
        item_noun=item_noun,
        domain_description=domain_description,
        constraint_list=constraint_list,
    )

    try:
        data = call_llm_json(
            prompt,
            model,
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning("LLM realism-weight call failed; using defaults", exc_info=True)
        return {c.name: (3, False) for c in constraints if not c.desc_constraint}

    result: dict[str, tuple[int, bool]] = {}
    for item in data.get("constraints", []):
        name = item.get("name", "")
        weight = item.get("realism_weight")
        always = item.get("always_include", False)
        if name and isinstance(weight, (int, float)) and 1 <= weight <= 5:
            result[name] = (int(weight), bool(always))

    # Fill missing constraints with default weight=2
    for c in constraints:
        if not c.desc_constraint and c.name not in result:
            result[c.name] = (2, False)

    return result


def _calibrate_probabilities(
    constraints: list[ConstraintSpec],
    realism_weights: dict[str, tuple[int, bool]],
    catalog: pd.DataFrame,
    target_mean_constraints: float = 4.0,
    target_zero_pool_rate: float = 0.12,
    n_validation_samples: int = 400,
    max_iterations: int = 5,
) -> list[ConstraintSpec]:
    """Stage B: convert realism weights to calibrated probabilities using empirical feedback.

    1. Compute per-constraint restrictiveness from the catalog.
    2. Scale realism weights → initial probabilities (sum ≈ target_mean).
    3. Run a mini calibration sweep; if zero-pool rate is too high, reduce
       probabilities for the most restrictive constraints and re-check.
    """
    from shared.filter import OPERATORS

    active = [c for c in constraints if not c.desc_constraint and c.sampling_values]
    if not active:
        return constraints

    # --- Step 1: Per-constraint restrictiveness (median hit-rate across values) ---
    restrictiveness: dict[str, float] = {}
    for c in active:
        if c.attribute not in catalog.columns:
            restrictiveness[c.name] = 0.5
            continue
        op_fn = OPERATORS.get(c.operator)
        if op_fn is None:
            restrictiveness[c.name] = 0.5
            continue
        col = catalog[c.attribute]
        rates = []
        for v in c.sampling_values:
            mask = op_fn(col, v)
            rates.append(float(mask.sum()) / max(len(catalog), 1))
        restrictiveness[c.name] = float(np.median(rates)) if rates else 0.5

    # --- Step 2: Convert weights to initial probabilities ---
    raw_weights: dict[str, float] = {}
    for c in active:
        w, always = realism_weights.get(c.name, (2, False))
        c.always_include = always

        med_rate = restrictiveness.get(c.name, 0.5)
        # Penalise constraints where most values match < 2% of catalog
        if med_rate < 0.02:
            w *= 0.4
        elif med_rate < 0.05:
            w *= 0.7
        raw_weights[c.name] = float(w)

    # Normalise so non-always-include weights sum to target_mean minus always-include count
    always_count = sum(1 for c in active if c.always_include)
    target_sum = max(target_mean_constraints - always_count, 1.0)
    non_always_total = sum(raw_weights[c.name] for c in active if not c.always_include)

    if non_always_total > 0:
        scale = target_sum / non_always_total
    else:
        scale = 1.0

    for c in active:
        if c.always_include:
            c.sampling_probability = 1.0
        else:
            c.sampling_probability = min(max(raw_weights[c.name] * scale, 0.05), 0.90)

    # --- Step 3: Mini calibration loop ---
    saved_state = random.getstate()

    for iteration in range(max_iterations):
        random.seed(99999 + iteration)
        n_zero = 0
        constraint_counts: list[int] = []

        for _ in range(n_validation_samples):
            sampled: dict[str, Any] = {}
            for c in active:
                if c.always_include or random.random() < c.sampling_probability:
                    sampled[c.name] = random.choice(c.sampling_values)
            constraint_counts.append(len(sampled))
            if sampled:
                pool = GenericFilter.apply(catalog, sampled, constraints)
                if len(pool) == 0:
                    n_zero += 1

        zero_rate = n_zero / n_validation_samples
        mean_c = float(np.mean(constraint_counts)) if constraint_counts else 0.0

        logger.info(
            "Probability calibration iter %d: zero_rate=%.2f (target≤%.2f) "
            "mean_constraints=%.1f (target=%.1f)",
            iteration, zero_rate, target_zero_pool_rate, mean_c, target_mean_constraints,
        )

        good_zero = zero_rate <= target_zero_pool_rate * 1.3
        good_mean = abs(mean_c - target_mean_constraints) < 0.8
        if good_zero and good_mean:
            break

        # Adjust: reduce probabilities for most-restrictive constraints
        if zero_rate > target_zero_pool_rate:
            sorted_by_restrict = sorted(
                [(c, restrictiveness.get(c.name, 1.0)) for c in active if not c.always_include],
                key=lambda x: x[1],
            )
            # Reduce the bottom half (most restrictive) more aggressively
            mid = len(sorted_by_restrict) // 2
            for i, (c, _) in enumerate(sorted_by_restrict):
                factor = 0.75 if i < mid else 0.90
                c.sampling_probability = max(c.sampling_probability * factor, 0.05)

        # Adjust mean constraint count
        if mean_c > target_mean_constraints + 0.8:
            ratio = target_mean_constraints / max(mean_c, 0.1)
            for c in active:
                if not c.always_include:
                    c.sampling_probability = max(c.sampling_probability * ratio, 0.05)
        elif mean_c < target_mean_constraints - 0.8:
            ratio = target_mean_constraints / max(mean_c, 0.1)
            for c in active:
                if not c.always_include:
                    c.sampling_probability = min(c.sampling_probability * ratio, 0.90)

    random.setstate(saved_state)

    logger.info(
        "Final probabilities: %s",
        {c.name: round(c.sampling_probability, 3) for c in active},
    )
    return constraints


def _refine_sampling_probabilities(
    constraints: list[ConstraintSpec],
    item_noun: str,
    domain_description: str,
    model: str = "gpt-4.1-mini",
    catalog: pd.DataFrame | None = None,
    target_mean_constraints: float = 4.0,
    target_zero_pool_rate: float = 0.12,
) -> list[ConstraintSpec]:
    """Two-stage probability refinement: LLM realism weights + empirical calibration.

    Stage A: LLM assigns 1-5 realism weights per constraint.
    Stage B: Weights are converted to calibrated probabilities using a
             mini calibration sweep against the catalog (when provided).

    Falls back to LLM-only probabilities (legacy behaviour) when catalog
    is not provided.
    """
    realism_weights = _get_realism_weights(constraints, item_noun, domain_description, model)

    # Apply always_include from LLM
    for c in constraints:
        if c.name in realism_weights:
            _, always = realism_weights[c.name]
            c.always_include = always

    if catalog is not None and len(catalog) > 0:
        constraints = _calibrate_probabilities(
            constraints, realism_weights, catalog,
            target_mean_constraints=target_mean_constraints,
            target_zero_pool_rate=target_zero_pool_rate,
        )
    else:
        # Legacy fallback: convert 1-5 weights to probabilities without empirical check
        active = [c for c in constraints if not c.desc_constraint and c.sampling_values]
        always_count = sum(1 for c in active if c.always_include)
        target_sum = max(target_mean_constraints - always_count, 1.0)
        total_w = sum(realism_weights.get(c.name, (2, False))[0] for c in active if not c.always_include)
        scale = target_sum / max(total_w, 1.0)

        for c in active:
            if c.always_include:
                c.sampling_probability = 1.0
            else:
                w = realism_weights.get(c.name, (2, False))[0]
                c.sampling_probability = min(max(w * scale, 0.05), 0.90)

    return constraints


# ---------------------------------------------------------------------------
# LLM-generated prompt fragments
# ---------------------------------------------------------------------------

def _coerce_llm_text_fragment(value: Any, default: str = "") -> str:
    """Normalize JSON from the LLM: bullet-style fields are sometimes returned as string lists."""
    if value is None:
        return default
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return "\n".join(parts) if parts else default
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_llm_plural(value: Any, fallback: str) -> str:
    """Plural noun occasionally arrives as a single-element list."""
    if value is None:
        return fallback
    if isinstance(value, list):
        for x in value:
            s = str(x).strip() if x is not None else ""
            if s:
                return s
        return fallback
    s = str(value).strip() if value else ""
    return s or fallback


def _generate_prompt_fragments(
    domain_description: str,
    item_noun: str,
    constraints: list[ConstraintSpec],
    model: str = "gpt-4.1-mini",
) -> PromptFragments:
    """Generate domain-specific prompt fragments via LLM."""
    constraint_names = ", ".join(c.name for c in constraints[:15])

    prompt = textwrap.dedent(f"""\
        You are configuring a conversational recommendation benchmark for {item_noun}s.

        DOMAIN: {domain_description}
        CONSTRAINT DIMENSIONS: {constraint_names}

        Generate domain-specific text fragments for the benchmark prompts. These
        will be injected into LLM prompts that generate simulated user profiles.

        Return JSON:
        {{
          "item_noun_plural": "the plural form of '{item_noun}' (e.g., 'laptops', 'knives', 'beauty products')",
          "system_persona": "a short persona for the recommendation agent (e.g., 'a laptop shopping assistant', 'a skincare advisor')",
          "use_case_description": "one phrase describing what the benchmark tests (e.g., 'laptop recommendation', 'beauty product selection')",
          "query_rules": "3-5 bullet points of domain-specific rules for the initial user query, e.g.:\\n- NO specific ingredient names\\n- NO brand names unless brand is a constraint\\n- YES to skin concerns and routines",
          "expertise_levels": {{
            "novice": "1-sentence description of what a novice {item_noun} shopper sounds like (language, confusion points)",
            "intermediate": "1-sentence description of what an intermediate {item_noun} shopper sounds like",
            "expert": "1-sentence description of what an expert {item_noun} shopper sounds like"
          }}
        }}
    """)

    try:
        data = call_llm_json(
            prompt,
            model,
            temperature=0.4,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning("LLM prompt fragment generation failed; using defaults", exc_info=True)
        return PromptFragments(
            domain_description=domain_description,
            item_noun=item_noun,
            item_noun_plural=item_noun + "s",
        )

    return PromptFragments(
        domain_description=domain_description,
        item_noun=item_noun,
        item_noun_plural=_coerce_llm_plural(
            data.get("item_noun_plural"), item_noun + "s"
        ),
        system_persona=_coerce_llm_text_fragment(data.get("system_persona"), ""),
        use_case_description=_coerce_llm_text_fragment(
            data.get("use_case_description"), ""
        ),
        query_rules=_coerce_llm_text_fragment(data.get("query_rules"), ""),
        expertise_levels=data.get("expertise_levels", {}),
    )


# ---------------------------------------------------------------------------
# Draft coherence module generation
# ---------------------------------------------------------------------------

def _generate_coherence_module(
    constraints: list[ConstraintSpec],
    domain_description: str,
    item_noun: str,
    output_dir: Path,
    model: str = "gpt-4.1-mini",
) -> str | None:
    """Generate a draft coherence module (Python) for the domain.

    Returns the dotted module path (e.g. 'domain.beauty.coherence') or None.
    """
    constraint_list = "\n".join(
        f"- {c.name}: {c.display_template} (operator: {c.operator.value})"
        for c in constraints
    )

    prompt = textwrap.dedent(f"""\
        You are writing coherence rules for a {item_noun} recommendation benchmark.

        DOMAIN: {domain_description}

        CONSTRAINTS (these are the dimensions users can specify):
        {constraint_list}

        Write a Python function `is_coherent(constraints: dict) -> bool` that rejects
        constraint COMBINATIONS that are physically impossible, logically contradictory,
        or that no realistic user would ever hold simultaneously.

        RULES FOR WRITING COHERENCE CHECKS:
        1. Only reject combinations that are genuinely contradictory, not merely rare.
        2. Each check should have a brief comment explaining WHY it's contradictory.
        3. For set-valued constraints on the same column, check for contradictions
           (e.g., requiring and excluding the same value).
        4. For numeric constraints, check for impossible ranges (min > max on same attribute).
        5. For cross-attribute contradictions, only flag combinations that are
           physically/logically impossible (not just uncommon).
        6. Return True if the combination is valid, False if contradictory.
        7. Access constraint values with constraints.get("constraint_name") — always
           check for None before comparing.

        Return ONLY the Python function body as a JSON string. No imports needed.

        Return JSON:
        {{
          "rules": [
            {{
              "description": "brief description of what this rule checks",
              "code": "python expression that evaluates to True when the combination IS contradictory/invalid (this will be placed in `if <code>: return False`)"
            }}
          ]
        }}
    """)

    try:
        data = call_llm_json(
            prompt,
            model,
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning("LLM coherence generation failed", exc_info=True)
        return None

    rules = data.get("rules", [])
    if not rules:
        logger.warning("LLM produced no coherence rules")
        return None

    lines = [
        '"""Auto-generated coherence rules — review before use."""',
        "",
        "",
        "def is_coherent(constraints: dict) -> bool:",
        '    """Reject constraint sets that are logically contradictory.',
        "",
        "    Auto-generated by config_generator. Review each rule and adjust",
        "    thresholds based on empirical pool-size analysis.",
        '    """',
    ]

    for rule in rules:
        desc = rule.get("description", "")
        code = rule.get("code", "")
        if not code:
            continue
        lines.append(f"    # {desc}")
        lines.append(f"    if {code}:")
        lines.append("        return False")
        lines.append("")

    lines.append("    return True")
    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    coherence_path = output_dir / "coherence.py"
    coherence_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Draft coherence module written to %s", coherence_path)

    return coherence_module_dotted_path(coherence_path)


def coherence_module_dotted_path(coherence_py: Path) -> str | None:
    """Dotted import path for ``coherence_py`` (must live under a ``domain`` path segment)."""
    parts = coherence_py.resolve().parts
    try:
        domain_idx = parts.index("domain")
        return ".".join(parts[domain_idx:]).replace(".py", "")
    except ValueError:
        return None


def resolve_coherence_module_path(output_dir: Path) -> str | None:
    """Return the config ``coherence_module`` string if ``output_dir/coherence.py`` exists."""
    p = output_dir / "coherence.py"
    if not p.is_file():
        return None
    return coherence_module_dotted_path(p)


def generate_coherence_module_for_domain(
    config: DomainConfig,
    output_dir: Path,
    *,
    model: str = "gpt-4.1-mini",
) -> str | None:
    """LLM-generate ``coherence.py`` and return the dotted module path (or None on failure).

    Typically run after :func:`generate_domain_config` so constraints and domain text
    match the saved config. Updates ``config.coherence_module`` only in the caller.
    """
    domain_desc = (config.prompt_fragments.domain_description or "").strip() or config.name
    return _generate_coherence_module(
        config.constraints,
        domain_desc,
        config.item_noun,
        output_dir,
        model,
    )


# ---------------------------------------------------------------------------
# Viability check for constraints
# ---------------------------------------------------------------------------

def _check_constraint_viability(constraints: list[ConstraintSpec]) -> list[ConstraintSpec]:
    """Disable constraints with no usable sampling values."""
    for c in constraints:
        if not c.sampling_values or len(c.sampling_values) == 0:
            logger.warning(
                "Constraint '%s' has no sampling values — setting probability to 0",
                c.name,
            )
            c.sampling_probability = 0.0
    return constraints


# ---------------------------------------------------------------------------
# Catalog-size-aware difficulty brackets
# ---------------------------------------------------------------------------

def _compute_difficulty_brackets(
    catalog_size: int,
    total_tasks: int = 250,
) -> dict[str, DifficultyBracket]:
    """Compute difficulty brackets scaled to catalog size.

    Brackets are defined as fractions of catalog size with absolute floors
    so they stay sensible for small catalogs.  Target counts follow the
    30 / 30 / 20 / 10 / 10 distribution.
    """
    N = max(catalog_size, 100)

    return {
        "small": DifficultyBracket(
            pool_range=(max(20, int(N * 0.002)), max(80, int(N * 0.010))),
            min_constraints=3,
            desc_constraint_probability=0.35,
            target_count=max(1, int(total_tasks * 0.30)),
        ),
        "medium": DifficultyBracket(
            pool_range=(max(8, int(N * 0.0005)), max(40, int(N * 0.003))),
            min_constraints=4,
            desc_constraint_probability=0.50,
            target_count=max(1, int(total_tasks * 0.30)),
        ),
        "large": DifficultyBracket(
            pool_range=(max(3, int(N * 0.0001)), max(15, int(N * 0.001))),
            min_constraints=5,
            desc_constraint_probability=0.65,
            target_count=max(1, int(total_tasks * 0.20)),
        ),
        "oc_feasible": DifficultyBracket(
            pool_range=(1, 3),
            min_constraints=4,
            desc_constraint_probability=0.15,
            target_count=max(1, int(total_tasks * 0.10)),
        ),
        "oc_infeasible": DifficultyBracket(
            pool_range=(0, 0),
            min_constraints=4,
            desc_constraint_probability=0.15,
            target_count=max(1, int(total_tasks * 0.10)),
        ),
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_domain_config(
    triage: TriageResult,
    profile_data: dict[str, Any],
    *,
    catalog_path: str = "",
    model: str = "gpt-4.1-mini",
    catalog: pd.DataFrame | None = None,
    total_tasks: int = 250,
) -> DomainConfig:
    """Generate a draft DomainConfig from triage result and profile data.

    Args:
        triage: Column triage result.
        profile_data: Profiler output dict.
        catalog_path: Path to the catalog file.
        model: LLM model for generation steps.
        catalog: Optional catalog DataFrame.  When provided, enables
            empirical probability calibration and catalog-size-aware difficulty
            brackets.
        total_tasks: Target total task count for bracket sizing
            (default 250, split 30/30/20/10/10).

    Coherence rules are not generated here; use
    :func:`generate_coherence_module_for_domain` after saving the config.

    The output needs human review — especially trigger keywords and sampling
    value ranges (and coherence rules, when generated separately).
    """
    col_profiles = {c["name"]: c for c in profile_data.get("columns", [])}
    n_rows = profile_data.get("n_rows_profiled", 0)

    attributes: list[AttributeSpec] = []
    constraints: list[ConstraintSpec] = []

    for col in triage.columns:
        if col.role == ColumnRole.DROP:
            continue

        prof = col_profiles.get(col.column) or {}
        if prof.get("inferred_kind"):
            col.inferred_kind = prof["inferred_kind"]

        # Type-role guard: downgrade unsupported kind+role combinations
        col = _apply_type_role_guard(col)

        attributes.append(_make_attribute(col))

        if col.role in (ColumnRole.HARD_FILTER, ColumnRole.SET_FILTER):
            col_profile = col_profiles.get(col.column, {})
            constraints.extend(_make_constraint(col, col_profile, n_rows))

    if not (triage.domain or "").strip():
        raise ValueError(
            "TriageResult.domain is empty; pass domain_description=... to triage_columns().",
        )

    constraints = _check_constraint_viability(constraints)

    # --- Two-stage probability refinement ---
    constraints = _refine_sampling_probabilities(
        constraints, triage.item_noun, triage.domain, model,
        catalog=catalog,
    )

    preference_attributes = _generate_preference_attributes(triage)

    triggers, violation_triggers = _generate_triggers(
        constraints, triage.item_noun,
        domain_description=triage.domain,
        preference_attributes=preference_attributes,
        model=model,
    )

    prompt_fragments = _generate_prompt_fragments(
        triage.domain, triage.item_noun, constraints, model,
    )

    # --- Difficulty brackets (catalog-size-aware) ---
    catalog_size = len(catalog) if catalog is not None else n_rows
    difficulty = _compute_difficulty_brackets(catalog_size, total_tasks)

    return DomainConfig(
        name=triage.item_noun.lower().replace(" ", "_"),
        item_noun=triage.item_noun,
        item_noun_plural=prompt_fragments.item_noun_plural,
        catalog_path=catalog_path,
        id_column=triage.id_column,
        attributes=attributes,
        constraints=constraints,
        triggers=triggers,
        violation_triggers=violation_triggers,
        preference_attributes=preference_attributes,
        difficulty=difficulty,
        prompt_fragments=prompt_fragments,
        coherence_module=None,
    )
