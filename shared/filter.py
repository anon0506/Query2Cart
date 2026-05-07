"""Domain-agnostic catalog filtering engine.

``GenericFilter.apply()`` replaces every hardcoded ``filter()`` and
``filter_by_hard_constraints()`` call in the codebase.  It reads operator
semantics from the ``ConstraintSpec`` registry, so adding a new constraint
type never touches this file — only the domain config.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from shared.config import ConstraintOp, ConstraintSpec


def _normalize_set_targets(val: Any) -> list[str]:
    """Turn constraint values into a list of tag strings for set operators.

    Sampling stores a *single* string (e.g. one OFF tag). Code must not use
    ``for x in that_string`` (that iterates characters). Lists/tuples keep
    multiple targets.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(v).lower() for v in val]
    if isinstance(val, str):
        return [val.lower()]
    if isinstance(val, np.ndarray):
        flat = val.tolist()
        if isinstance(flat, list):
            return [str(v).lower() for v in flat]
        return [str(flat).lower()]
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return [str(v).lower() for v in val]
    return [str(val).lower()]


def _cell_to_tag_list(x: Any) -> list[str] | None:
    """Row value for a set-valued column as a list of lowercase tag strings."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (str, bytes)):
        return None
    if isinstance(x, (list, tuple)):
        return [str(i).lower() for i in x]
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return []
        flat = x.tolist()
        if isinstance(flat, list):
            return [str(i).lower() for i in flat]
        return [str(flat).lower()]
    return None


def _coerce_series_datetime_utc(col: pd.Series) -> pd.Series | None:
    """If *col* holds temporal values, return UTC-normalized datetimes; else None."""
    if pd.api.types.is_datetime64_any_dtype(col):
        s = pd.to_datetime(col, utc=True, errors="coerce")
        return s if s.notna().any() else None
    if col.dtype != object:
        return None
    head = col.dropna()
    if head.empty:
        return None
    v = head.iloc[0]
    if isinstance(v, (str, bytes)):
        return None
    import datetime as dt

    if isinstance(v, (dt.date, dt.datetime)):
        s = pd.to_datetime(col, utc=True, errors="coerce")
        return s if s.notna().any() else None
    if type(v).__name__ == "Timestamp":
        s = pd.to_datetime(col, utc=True, errors="coerce")
        return s if s.notna().any() else None
    return None


def _parse_constraint_datetime_bound(val: Any) -> pd.Timestamp:
    if isinstance(val, pd.Timestamp):
        t = val
    elif isinstance(val, (int, float)) and not isinstance(val, bool):
        t = pd.Timestamp(float(val), unit="s", tz="UTC")
    else:
        t = pd.Timestamp(val)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _op_lte(col: pd.Series, val: Any) -> pd.Series:
    tcol = _coerce_series_datetime_utc(col)
    if tcol is not None:
        try:
            bound = _parse_constraint_datetime_bound(val)
        except (ValueError, TypeError, OSError):
            return pd.Series(False, index=col.index)
        return tcol.notna() & (tcol <= bound)
    col = pd.to_numeric(col, errors="coerce")
    return col.notna() & (col <= float(val))

def _op_gte(col: pd.Series, val: Any) -> pd.Series:
    tcol = _coerce_series_datetime_utc(col)
    if tcol is not None:
        try:
            bound = _parse_constraint_datetime_bound(val)
        except (ValueError, TypeError, OSError):
            return pd.Series(False, index=col.index)
        return tcol.notna() & (tcol >= bound)
    col = pd.to_numeric(col, errors="coerce")
    return col.notna() & (col >= float(val))

def _op_eq(col: pd.Series, val: Any) -> pd.Series:
    return col.str.lower().fillna("") == str(val).lower()

def _op_eq_any(col: pd.Series, val: Any) -> pd.Series:
    targets = _normalize_set_targets(val)
    return col.str.lower().fillna("").isin(targets)

def _op_neq(col: pd.Series, val: Any) -> pd.Series:
    return col.str.lower().fillna("") != str(val).lower()

def _op_in_set(col: pd.Series, val: Any) -> pd.Series:
    lowered = _normalize_set_targets(val)
    return col.str.lower().fillna("").isin(lowered)

def _op_contains(col: pd.Series, val: Any) -> pd.Series:
    target = str(val).lower()
    def _check(x: Any) -> bool:
        items = _cell_to_tag_list(x)
        if items is None:
            return False
        return target in items
    return col.apply(_check)

def _op_contains_all(col: pd.Series, val: Any) -> pd.Series:
    targets = _normalize_set_targets(val)
    if not targets:
        return pd.Series(True, index=col.index)
    def _check(x: Any) -> bool:
        items = _cell_to_tag_list(x)
        if items is None:
            return False
        return all(t in items for t in targets)
    return col.apply(_check)

def _op_contains_any(col: pd.Series, val: Any) -> pd.Series:
    targets = _normalize_set_targets(val)
    if not targets:
        return pd.Series(True, index=col.index)
    def _check(x: Any) -> bool:
        items = _cell_to_tag_list(x)
        if items is None:
            return False
        return any(t in items for t in targets)
    return col.apply(_check)

def _op_not_contains(col: pd.Series, val: Any) -> pd.Series:
    target = str(val).lower()
    def _check(x: Any) -> bool:
        items = _cell_to_tag_list(x)
        if items is None:
            return False
        return target not in items
    return col.apply(_check)

def _op_not_contains_any(col: pd.Series, val: Any) -> pd.Series:
    targets = _normalize_set_targets(val)
    if not targets:
        return pd.Series(True, index=col.index)
    def _check(x: Any) -> bool:
        items = _cell_to_tag_list(x)
        if items is None:
            return False
        return not any(t in items for t in targets)
    return col.apply(_check)

def _op_not_contains_all(col: pd.Series, val: Any) -> pd.Series:
    targets = _normalize_set_targets(val)
    if not targets:
        return pd.Series(True, index=col.index)
    def _check(x: Any) -> bool:
        items = _cell_to_tag_list(x)
        if items is None:
            return False
        return not all(t in items for t in targets)
    return col.apply(_check)


def _op_substring(col: pd.Series, val: Any) -> pd.Series:
    return col.str.lower().fillna("").str.contains(str(val).lower(), regex=False)

def _op_boolean(col: pd.Series, val: Any) -> pd.Series:
    target = bool(val) if isinstance(val, bool) else str(val).lower() in ("true", "1", "yes")
    return col.fillna(False) == target

def _op_range(col: pd.Series, val: Any) -> pd.Series:
    lo, hi = val
    col = pd.to_numeric(col, errors="coerce")
    return col.notna() & (col >= float(lo)) & (col <= float(hi))


def hard_constraint_value_missing(val: Any) -> bool:
    """True if *val* should be treated as missing for hard constraint checking.

    Aligns with :meth:`GenericFilter.apply`, which excludes null/NA cells from
    the constraint-satisfying pool (operators require ``notna`` or non-empty
    tags). String sentinels like ``NA`` / ``N/A`` are treated as missing.
    """
    if val is None:
        return True
    if isinstance(val, str):
        st = val.strip()
        return st.upper() in ("NA", "N/A", "NAN")
    if isinstance(val, bytes):
        return hard_constraint_value_missing(val.decode("utf-8", errors="replace"))
    if pd.api.types.is_scalar(val):
        return bool(pd.isna(val))
    if isinstance(val, (list, tuple)):
        return len(val) == 0
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return True
        m = pd.isna(val)
        return bool(getattr(m, "all", lambda: False)())
    return False


OPERATORS: dict[ConstraintOp, Callable[[pd.Series, Any], pd.Series]] = {
    ConstraintOp.LTE: _op_lte,
    ConstraintOp.GTE: _op_gte,
    ConstraintOp.EQ: _op_eq,
    ConstraintOp.EQ_ANY: _op_eq_any,
    ConstraintOp.NEQ: _op_neq,
    ConstraintOp.IN_SET: _op_in_set,
    ConstraintOp.CONTAINS: _op_contains,
    ConstraintOp.CONTAINS_ALL: _op_contains_all,
    ConstraintOp.CONTAINS_ANY: _op_contains_any,
    ConstraintOp.NOT_CONTAINS: _op_not_contains,
    ConstraintOp.NOT_CONTAINS_ANY: _op_not_contains_any,
    ConstraintOp.NOT_CONTAINS_ALL: _op_not_contains_all,
    ConstraintOp.SUBSTRING: _op_substring,
    ConstraintOp.BOOLEAN: _op_boolean,
    ConstraintOp.RANGE: _op_range,
}


class GenericFilter:
    """Domain-agnostic catalog filtering using a constraint registry."""

    @staticmethod
    def apply(
        catalog: pd.DataFrame,
        constraints: dict[str, Any],
        constraint_registry: list[ConstraintSpec],
    ) -> pd.DataFrame:
        """Filter *catalog* by *constraints* using operators declared in the registry.

        Items with null on a constrained attribute are **excluded** (conservative
        approach, matching the existing laptop pipeline behaviour).

        Args:
            catalog: Full catalog DataFrame.
            constraints: ``{constraint_name: value}`` dict (e.g. from a task).
            constraint_registry: List of ``ConstraintSpec`` from the domain config.

        Returns:
            Filtered DataFrame containing only rows satisfying every constraint.
        """
        registry_by_name = {c.name: c for c in constraint_registry}
        mask = pd.Series(True, index=catalog.index)

        for constraint_name, value in constraints.items():
            spec = registry_by_name.get(constraint_name)
            if spec is None:
                continue

            if spec.attribute not in catalog.columns:
                continue

            col = catalog[spec.attribute]
            op_fn = OPERATORS.get(spec.operator)
            if op_fn is None:
                raise ValueError(f"Unknown operator {spec.operator!r} for constraint {constraint_name!r}")

            mask &= op_fn(col, value)

        return catalog[mask]

    @staticmethod
    def check_violations(
        product: dict | pd.Series,
        constraints: dict[str, Any],
        constraint_registry: list[ConstraintSpec],
    ) -> list[str]:
        """Check whether a single product satisfies all constraints.

        Returns a list of human-readable violation strings (empty = all satisfied).
        """
        registry_by_name = {c.name: c for c in constraint_registry}
        violations: list[str] = []

        for constraint_name, value in constraints.items():
            spec = registry_by_name.get(constraint_name)
            if spec is None:
                continue

            actual = product.get(spec.attribute) if isinstance(product, dict) else product.get(spec.attribute)
            if hard_constraint_value_missing(actual):
                violations.append(
                    f"{spec.attribute} is missing or NA "
                    f"(cannot satisfy constraint '{constraint_name}')"
                )
                continue

            single_row = pd.DataFrame([{spec.attribute: actual}])
            op_fn = OPERATORS.get(spec.operator)
            if op_fn is None:
                continue

            result = op_fn(single_row[spec.attribute], value)
            if not result.iloc[0]:
                label = spec.display_template.format(value=value)
                violations.append(f"{spec.attribute}={actual} violates {label}")

        return violations
