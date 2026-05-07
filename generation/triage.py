"""Column triage: classify profiler output into domain roles.

Takes the output of ``dataset_profile.profile_dataset()`` and assigns each
column a role (id, hard_filter, soft_preference, set_filter, embedding_text,
drop).  Two-phase approach:

Phase 1 — Filter: expanded heuristics + LLM keep/drop decision.
Phase 2 — Classify: LLM assigns roles to retained columns only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from generation.profile import format_retained_columns_text
from shared.llm import call_llm_json, format_prompt_template, resolve_prompt_path

logger = logging.getLogger(__name__)


class ColumnRole(str, Enum):
    ID = "id"
    HARD_FILTER = "hard_filter"
    SOFT_PREFERENCE = "soft_preference"
    SET_FILTER = "set_filter"
    EMBEDDING_TEXT = "embedding_text"
    DROP = "drop"


@dataclass
class ColumnAssignment:
    column: str
    role: ColumnRole
    inferred_kind: str
    null_rate: float
    n_unique: int
    user_facing_name: str = ""
    attribute_description: str = ""     # what the column encodes in this catalog
    constraint_direction: str = ""      # max | min (for numeric hard_filter)
    preference_direction: str = ""      # minimize | maximize | match
    membership_type: str = ""           # contains_any | contains_all (for set_filter)
    constraint_polarity: str = ""       # positive | negative (for set_filter: include vs exclude)
    note: str = ""
    auto_assigned: bool = True


@dataclass
class TriageResult:
    domain: str
    source: str
    item_noun: str
    profiled_rows: int
    id_column: str
    columns: list[ColumnAssignment] = field(default_factory=list)

    def by_role(self, role: ColumnRole) -> list[ColumnAssignment]:
        return [c for c in self.columns if c.role == role]

    def save(self, path: str) -> None:
        from dataclasses import asdict
        data = asdict(self)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> "TriageResult":
        raw = json.loads(Path(path).read_text())
        def _one_column(c: dict[str, Any]) -> ColumnAssignment:
            merged = {**c, "role": ColumnRole(c["role"])}
            merged.setdefault("attribute_description", "")
            merged.pop("needs_structured_value_extraction", None)
            return ColumnAssignment(**merged)
        raw["columns"] = [_one_column(c) for c in raw["columns"]]
        return cls(**raw)


# ---------------------------------------------------------------------------
# Heuristic rules (expanded)
# ---------------------------------------------------------------------------

_AUDIT_TIMESTAMP_PATTERNS = re.compile(
    r"(^|_)(last_modified|last_updated|created|updated|modified|entry_date)"
    r"(_t|_at|_ts|_date|_time)?$",
    re.IGNORECASE,
)

_SCHEMA_METADATA_NAMES = frozenset({
    "schema_version", "rev", "revision", "version",
    "data_schema_version", "api_version",
})


def _apply_heuristic_rules(
    col_summary: dict[str, Any],
    all_summaries: list[dict[str, Any]],
    noise_columns: dict[str, str],
    null_rate_threshold: float,
) -> ColumnAssignment | None:
    """Return an assignment if heuristic rules are conclusive, else None."""
    name = col_summary["name"]
    kind = col_summary["inferred_kind"]
    null_rate = col_summary["null_rate"]
    n_unique = col_summary["n_unique"]
    n_analyzed = col_summary["n_analyzed"]
    tags = col_summary.get("quality_tags", [])
    details = col_summary.get("details", {})

    base = dict(
        column=name,
        inferred_kind=kind,
        null_rate=null_rate,
        n_unique=n_unique,
        auto_assigned=True,
    )

    # 1. All null
    if kind == "all_null":
        return ColumnAssignment(**base, role=ColumnRole.DROP, note="all null")

    # 2. Constant
    if n_analyzed > 0 and n_unique == 1:
        return ColumnAssignment(**base, role=ColumnRole.DROP, note="constant column")

    # 3. Profiler noise_columns dict
    if name in noise_columns:
        return ColumnAssignment(
            **base, role=ColumnRole.DROP,
            note=f"profiler noise: {noise_columns[name]}",
        )

    # 4. Noise quality tags
    noise_tags = [t for t in tags if t.startswith("noise_")]
    if noise_tags:
        return ColumnAssignment(
            **base, role=ColumnRole.DROP,
            note=f"noise tag: {noise_tags[0]}",
        )

    # 5. Sparse (tightened threshold, default 0.70) — exempt text columns
    if null_rate > null_rate_threshold and kind != "string_long_text":
        return ColumnAssignment(
            **base, role=ColumnRole.DROP,
            note=f"sparse (null_rate={null_rate:.0%}, threshold={null_rate_threshold:.0%})",
        )

    # 6. Near-constant boolean
    if kind == "bool":
        true_rate = details.get("true_rate")
        if true_rate is not None and (true_rate < 0.05 or true_rate > 0.95):
            return ColumnAssignment(
                **base, role=ColumnRole.DROP,
                note=f"near-constant boolean (true_rate={true_rate:.1%})",
            )

    # 7. Audit timestamp
    if (
        _AUDIT_TIMESTAMP_PATTERNS.search(name)
        and kind in ("int_numeric", "float_numeric")
    ):
        min_val = details.get("min", 0)
        if min_val and min_val > 1e9:
            return ColumnAssignment(
                **base, role=ColumnRole.DROP,
                note="audit timestamp (epoch values)",
            )

    # 8. Schema / revision metadata
    if name.lower() in _SCHEMA_METADATA_NAMES:
        return ColumnAssignment(
            **base, role=ColumnRole.DROP,
            note="system schema/revision metadata",
        )

    # 9. Redundant count-of-list (X_n where X or X_tags is a list column)
    if name.endswith("_n") and kind in ("int_numeric", "float_numeric"):
        stem = name[:-2]
        list_sibling = any(
            s["name"] in (stem, f"{stem}_tags")
            and s["inferred_kind"] == "list_of_values"
            for s in all_summaries
        )
        if list_sibling:
            return ColumnAssignment(
                **base, role=ColumnRole.DROP,
                note=f"redundant count of list column '{stem}' or '{stem}_tags'",
            )

    return None


# ---------------------------------------------------------------------------
# Phase 1: Filter columns (keep vs. drop)
# ---------------------------------------------------------------------------


def _parse_phase1_retain(raw: Any) -> tuple[set[str], dict[str, str]]:
    """Parse LLM ``retain`` as a list of column name strings and/or ``{column, reason}`` objects."""
    if not raw or not isinstance(raw, list):
        return set(), {}
    names: set[str] = set()
    reasons: dict[str, str] = {}
    for item in raw:
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
        elif isinstance(item, dict):
            col = (item.get("column") or "").strip()
            if not col:
                continue
            names.add(col)
            r = (item.get("reason") or "").strip()
            if r:
                reasons[col] = r
    return names, reasons


def _build_phase1_batch(unassigned: list[dict[str, Any]]) -> str:
    """Format unassigned columns for the Phase 1 filtering prompt.

    Reuses the same format as ``print_retained_columns_report`` so the
    LLM sees the same rich per-column detail the user reviews.
    """
    return format_retained_columns_text(unassigned)


def _phase1_filter_columns(
    columns_data: list[dict[str, Any]],
    noise_columns: dict[str, str],
    domain_description: str,
    item_noun: str,
    model: str,
    null_rate_threshold: float,
) -> tuple[list[ColumnAssignment], list[dict[str, Any]], dict[str, str]]:
    """Phase 1: decide keep vs. drop for each column.

    Returns (auto_assignments, retained_column_summaries, phase1_retain_reasons).
    """
    assigned: list[ColumnAssignment] = []
    unassigned: list[dict[str, Any]] = []

    for col in columns_data:
        result = _apply_heuristic_rules(
            col, columns_data, noise_columns, null_rate_threshold,
        )
        if result is not None:
            assigned.append(result)
        else:
            unassigned.append(col)

    logger.info(
        "Phase 1 heuristics: %d auto-dropped, %d sent to LLM",
        len(assigned),
        len(unassigned),
    )

    if not unassigned:
        return assigned, [], {}

    batch_text = _build_phase1_batch(unassigned)
    template = Path(resolve_prompt_path("triage_phase1_filter.txt")).read_text(
        encoding="utf-8",
    )
    user_prompt = format_prompt_template(
        template,
        domain_description=domain_description,
        item_noun=item_noun,
        column_summaries=batch_text,
    )

    try:
        llm_result = call_llm_json(
            user_prompt,
            model,
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning(
            "Phase 1 LLM filtering failed; retaining all %d heuristic-ambiguous columns",
            len(unassigned),
            exc_info=True,
        )
        return assigned, unassigned, {}

    retain_set, phase1_retain_reasons = _parse_phase1_retain(
        llm_result.get("retain"),
    )
    drop_list = llm_result.get("drop", [])
    drop_reasons = {
        item["column"]: item.get("reason", "LLM decision")
        for item in drop_list
        if isinstance(item, dict) and "column" in item
    }

    retained_summaries: list[dict[str, Any]] = []
    for col in unassigned:
        col_name = col["name"]
        if col_name in retain_set:
            retained_summaries.append(col)
        elif col_name in drop_reasons:
            assigned.append(ColumnAssignment(
                column=col_name,
                role=ColumnRole.DROP,
                inferred_kind=col["inferred_kind"],
                null_rate=col["null_rate"],
                n_unique=col["n_unique"],
                note=f"Phase 1 LLM: {drop_reasons[col_name]}",
                auto_assigned=False,
            ))
        else:
            retained_summaries.append(col)

    n_retained = len(retained_summaries)
    if n_retained < 6:
        logger.warning("Phase 1 retained only %d columns — check data quality or prompt", n_retained)
    elif n_retained > 30:
        logger.info("Phase 1 retained %d columns — Phase 2 prompt will be substantial", n_retained)
    else:
        logger.info("Phase 1 retained %d columns for role assignment", n_retained)

    return assigned, retained_summaries, phase1_retain_reasons


# ---------------------------------------------------------------------------
# Phase 2: Assign roles and metadata to retained columns
# ---------------------------------------------------------------------------

def _build_phase2_batch(retained: list[dict[str, Any]]) -> str:
    """Format retained columns for the Phase 2 role-assignment prompt."""
    lines = []
    for col in retained:
        details = col.get("details", {})
        summary = col.get("column_summary") or ""
        tags = ", ".join(
            t for t in col.get("quality_tags", []) if not t.startswith("noise_")
        )
        sibs = details.get("possible_sibling_columns")
        sib_line = ""
        if sibs:
            sib_line = (
                f"  Related columns (string-prefix relationship — keep the best for filtering, "
                f"not necessarily all): {', '.join(sibs)}"
            )

        parts = [
            f"- {col['name']}  kind={col['inferred_kind']}  "
            f"null={col['null_rate']:.0%}  unique={col['n_unique']}  "
            f"tags=[{tags}]",
        ]
        if sib_line:
            parts.append(sib_line)

        if col["inferred_kind"] in ("int_numeric", "float_numeric", "datetime"):
            stats = []
            for key in ("min", "max", "mean", "p5", "p50", "p95"):
                val = details.get(key)
                if val is not None:
                    stats.append(f"{key}={val}")
            if stats:
                parts.append(f"    stats: {', '.join(stats)}")

        top_k = details.get("top_10_value_frequencies") or details.get("top_10_element_frequencies", [])
        if top_k:
            top_vals = ", ".join(
                f"{item['value']} ({item.get('share', 0):.1%})"
                for item in top_k[:5]
            )
            parts.append(f"    top values: {top_vals}")

        if summary:
            parts.append(
                f"    Brief description of this attribute (from profiling): {summary}",
            )
        else:
            parts.append(
                "    Brief description of this attribute: (none) — infer what this field "
                "means from the column name, kind, and stats when writing "
                "attribute_description.",
            )

        lines.append("\n".join(parts))

    return "\n\n".join(lines)


def _merge_phase1_and_phase2_note(
    phase1_retain: str, phase2_note: str,
) -> str:
    p1 = phase1_retain.strip()
    p2 = (phase2_note or "").strip()
    if p1 and p2:
        return f"Phase 1 retain: {p1}. {p2}"
    if p1:
        return f"Phase 1 retain: {p1}"
    return p2


def _phase2_assign_roles(
    retained_summaries: list[dict[str, Any]],
    domain_description: str,
    item_noun: str,
    model: str,
    phase1_retain_reasons: dict[str, str] | None = None,
) -> list[ColumnAssignment]:
    """Phase 2: assign roles and metadata to retained columns."""
    if not retained_summaries:
        return []

    batch_text = _build_phase2_batch(retained_summaries)
    template = Path(resolve_prompt_path("triage_phase2_roles.txt")).read_text(
        encoding="utf-8",
    )
    user_prompt = format_prompt_template(
        template,
        domain_description=domain_description,
        item_noun=item_noun,
        n_columns=str(len(retained_summaries)),
        column_summaries=batch_text,
    )

    try:
        llm_result = call_llm_json(
            user_prompt,
            model,
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning(
            "Phase 2 LLM role assignment failed; defaulting %d columns to drop",
            len(retained_summaries),
            exc_info=True,
        )
        return [
            ColumnAssignment(
                column=col["name"],
                role=ColumnRole.DROP,
                inferred_kind=col["inferred_kind"],
                null_rate=col["null_rate"],
                n_unique=col["n_unique"],
                note=_merge_phase1_and_phase2_note(
                    (phase1_retain_reasons or {}).get(col["name"], ""),
                    "Phase 2 LLM failed; defaulted to drop",
                ),
                auto_assigned=True,
            )
            for col in retained_summaries
        ]

    col_lookup = {c["name"]: c for c in retained_summaries}
    p1_reasons = phase1_retain_reasons or {}
    assignments: list[ColumnAssignment] = []

    for item in llm_result.get("columns", []):
        col_name = item.get("column", "")
        col_data = col_lookup.get(col_name)
        if col_data is None:
            continue

        role_str = item.get("role", "drop")
        try:
            role = ColumnRole(role_str)
        except ValueError:
            role = ColumnRole.DROP

        assignments.append(ColumnAssignment(
            column=col_name,
            role=role,
            inferred_kind=col_data["inferred_kind"],
            null_rate=col_data["null_rate"],
            n_unique=col_data["n_unique"],
            user_facing_name=item.get("user_facing_name", ""),
            attribute_description=(item.get("attribute_description") or "").strip(),
            constraint_direction=item.get("constraint_direction", ""),
            preference_direction=item.get("preference_direction", ""),
            membership_type=item.get("membership_type", ""),
            constraint_polarity=item.get("constraint_polarity", ""),
            note=_merge_phase1_and_phase2_note(
                p1_reasons.get(col_name, ""), item.get("note", ""),
            ),
            auto_assigned=False,
        ))
        col_lookup.pop(col_name, None)

    for col_name, col_data in col_lookup.items():
        logger.warning("Phase 2 LLM did not assign column '%s'; defaulting to drop", col_name)
        assignments.append(ColumnAssignment(
            column=col_name,
            role=ColumnRole.DROP,
            inferred_kind=col_data["inferred_kind"],
            null_rate=col_data["null_rate"],
            n_unique=col_data["n_unique"],
            note=_merge_phase1_and_phase2_note(
                p1_reasons.get(col_name, ""),
                "not returned by Phase 2 LLM; defaulted to drop",
            ),
            auto_assigned=True,
        ))

    return assignments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triage_columns(
    profile_data: dict[str, Any],
    *,
    domain_description: str,
    item_noun: str,
    source_name: str = "unknown",
    model: str = "gpt-4.1-mini",
    null_rate_threshold: float = 0.70,
) -> TriageResult:
    """Run two-phase column triage on profiler output.

    Phase 1: expanded heuristics + LLM keep/drop filtering.
    Phase 2: LLM role and metadata assignment for retained columns.

    Returns a ``TriageResult`` for human review.
    """
    if not (domain_description or "").strip():
        raise ValueError(
            "domain_description is required (brief product / catalog context for the LLM)."
        )

    columns_data = profile_data.get("columns", [])
    noise_columns: dict[str, str] = profile_data.get("noise_columns", {})
    n_rows = profile_data.get("n_rows_profiled", 0)

    # Phase 1: filter
    auto_assignments, retained_summaries, phase1_retain_reasons = (
        _phase1_filter_columns(
            columns_data, noise_columns, domain_description, item_noun,
            model, null_rate_threshold,
        )
    )

    # Phase 2: classify retained columns
    llm_assignments = _phase2_assign_roles(
        retained_summaries, domain_description, item_noun, model,
        phase1_retain_reasons=phase1_retain_reasons,
    )

    all_assignments = auto_assignments + llm_assignments
    id_cols = [a for a in all_assignments if a.role == ColumnRole.ID]
    id_column = id_cols[0].column if id_cols else ""

    if not id_column:
        id_column = _fallback_id_detection(all_assignments, columns_data, n_rows)
        if id_column:
            for a in all_assignments:
                if a.column == id_column:
                    a.role = ColumnRole.ID
                    a.note = (a.note + "; " if a.note else "") + "promoted to ID by uniqueness fallback"
                    break

    return TriageResult(
        domain=domain_description,
        source=source_name,
        item_noun=item_noun,
        profiled_rows=n_rows,
        id_column=id_column,
        columns=all_assignments,
    )


def _fallback_id_detection(
    assignments: list[ColumnAssignment],
    columns_data: list[dict[str, Any]],
    n_rows: int,
) -> str:
    """Find an ID column when heuristic name-matching failed.

    Looks for any column with >= 99% unique values that is a string or int
    type (not float, not list). Prefers columns with higher uniqueness ratio.
    If nothing qualifies, returns empty string — caller should synthesize.
    """
    if n_rows == 0:
        return ""

    col_data_by_name = {c["name"]: c for c in columns_data}
    candidates: list[tuple[str, float]] = []

    for a in assignments:
        if a.role == ColumnRole.ID:
            continue
        if a.inferred_kind not in ("string_categorical", "int_numeric"):
            continue
        ratio = a.n_unique / n_rows if n_rows > 0 else 0.0
        if ratio >= 0.99:
            candidates.append((a.column, ratio))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: -x[1])
    best = candidates[0][0]
    logger.info("ID fallback: promoting column '%s' to ID (uniqueness ratio %.4f)", best, candidates[0][1])
    return best
