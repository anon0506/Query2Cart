"""Continuous preference-utility scoring for Query2Cart.

Products are scored on [0, 1] by how well they satisfy the user's
attribute preferences.  The full reward decomposes into:

    reward = I(all hard constraints met) * (0.5 + 0.5 * preference_utility)

Pool-based min-max normalization ensures utility is relative to the
best achievable option among constraint-satisfying products.  Geometric
weights allow trade-offs while respecting the user's priority ordering.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _utility_value_missing(val: Any) -> bool:
    """True if ``val`` should be treated as missing for utility scoring.

    ``pd.isna`` on array-likes returns an array, which must not be used in
    ``if`` directly.  Catalog cells may be numpy arrays (e.g. set-valued
    attributes stored as lists/arrays in parquet).
    """
    if val is None:
        return True
    if pd.api.types.is_scalar(val):
        return bool(pd.isna(val))
    m = pd.isna(val)
    if hasattr(m, "all"):
        return bool(m.all())
    return bool(m)


def _preference_series_as_float(series: pd.Series) -> pd.Series:
    """Coerce a catalog column to floats for pool min/max (numeric or parseable dates)."""
    v = series.dropna()
    if v.empty:
        return pd.Series(dtype=np.float64)
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().any():
        return num.dropna().astype(np.float64)
    dt = pd.to_datetime(v, errors="coerce", utc=True)
    if dt.notna().any():
        ok = dt[dt.notna()]
        return ok.astype("int64").astype(np.float64) / 1e9
    return pd.Series(dtype=np.float64)


def _scalar_to_preference_float(val: Any) -> float:
    """Coerce one cell the same way as :func:`_preference_series_as_float`."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return float("nan")
        return float(val.value) / 1e9
    if isinstance(val, datetime):
        return float(pd.Timestamp(val, tz="UTC").value) / 1e9
    if isinstance(val, date):
        return float(pd.Timestamp(datetime.combine(val, datetime.min.time()), tz="UTC").value) / 1e9
    if isinstance(val, np.datetime64):
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return float("nan")
        return float(ts.value) / 1e9
    if isinstance(val, (str, bytes)):
        s = str(val)
        n = pd.to_numeric(s, errors="coerce")
        if pd.notna(n):
            return float(n)
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.notna(dt):
            return float(dt.value) / 1e9
        return float("nan")
    if isinstance(val, (list, tuple, np.ndarray)):
        if isinstance(val, np.ndarray) and val.size == 0:
            return float("nan")
        first = val.ravel()[0] if isinstance(val, np.ndarray) else val[0]
        return _scalar_to_preference_float(first)
    try:
        return float(np.asarray(val, dtype=float).ravel()[0])
    except (TypeError, ValueError):
        return float("nan")


def _float_for_utility(val: Any) -> float:
    """Coerce a catalog cell to ``float`` for min/max utility (scalars or 0-d/1-d arrays)."""
    return _scalar_to_preference_float(val)



def compute_preference_weights(preferences: list[dict]) -> list[float]:
    """Geometric-decay weights from priority ordering.

    Priority 1 gets twice the weight of priority 2, which gets twice
    the weight of priority 3.  Weights sum to 1.

    Examples (sorted by priority):
        2 prefs -> [0.667, 0.333]
        3 prefs -> [0.571, 0.286, 0.143]
    """
    n = len(preferences)
    if n == 0:
        return []
    raw = [2.0 ** (n - i - 1) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def compute_pool_stats(
    pool: pd.DataFrame,
    preferences: list[dict],
) -> dict[str, dict[str, Any]]:
    """Min/max for each preference attribute across the constraint-satisfying pool.

    Returns ``{attribute_name: {"min": float, "max": float, "catalog_field": str}}``.
    """
    stats: dict[str, dict[str, Any]] = {}

    for pref in preferences:
        attr = pref["attribute"]
        direction = pref.get("direction", "maximize")
        catalog_field = attr

        if direction == "match":
            stats[attr] = {"min": 0.0, "max": 1.0, "catalog_field": catalog_field}
            continue

        if catalog_field not in pool.columns or len(pool) == 0:
            stats[attr] = {"min": 0.0, "max": 0.0, "catalog_field": catalog_field}
            continue

        coerced = _preference_series_as_float(pool[catalog_field])
        if len(coerced) == 0:
            stats[attr] = {"min": 0.0, "max": 0.0, "catalog_field": catalog_field}
            continue

        stats[attr] = {
            "min": float(coerced.min()),
            "max": float(coerced.max()),
            "catalog_field": catalog_field,
        }

    return stats


def compute_product_utility(
    product: pd.Series,
    preferences: list[dict],
    pool_stats: dict[str, dict[str, Any]],
    weights: list[float],
) -> float:
    """Score a single product on [0, 1] by attribute preferences.

    Per-attribute scoring:
      - minimize: ``(pool_max - value) / (pool_max - pool_min)``
      - maximize: ``(value - pool_min) / (pool_max - pool_min)``
      - match:    ``1.0`` if product matches target, else ``0.0``
      - degenerate (max == min): ``1.0``
      - missing value: ``0.0``

    Returns the weighted sum of per-attribute scores.
    """
    sorted_prefs = sorted(preferences, key=lambda p: p.get("priority", 99))

    if not sorted_prefs:
        return 1.0

    if len(sorted_prefs) != len(weights):
        return 0.0

    utility = 0.0
    for pref, weight in zip(sorted_prefs, weights):
        attr = pref["attribute"]
        direction = pref.get("direction", "maximize")
        stats = pool_stats.get(attr, {})
        catalog_field = stats.get("catalog_field", attr)

        if direction == "match":
            target = str(pref.get("target", "")).lower()
            actual = product.get(catalog_field)
            if _utility_value_missing(actual):
                score = 0.0
            elif isinstance(actual, (str, bytes)):
                score = 1.0 if str(actual).lower() == target else 0.0
            elif isinstance(actual, (list, tuple, np.ndarray)):
                if isinstance(actual, np.ndarray) and actual.size == 0:
                    score = 0.0
                else:
                    seq = actual.tolist() if isinstance(actual, np.ndarray) else list(actual)
                    tags = [str(x).lower() for x in seq]
                    score = 1.0 if target in tags else 0.0
            else:
                score = 1.0 if str(actual).lower() == target else 0.0
        else:
            val = product.get(catalog_field)
            if _utility_value_missing(val):
                score = 0.0
            else:
                val = _float_for_utility(val)
                if val is not val:  # NaN
                    score = 0.0
                else:
                    pmin = stats.get("min", 0.0)
                    pmax = stats.get("max", 0.0)

                    if pmax == pmin:
                        score = 1.0
                    elif direction == "minimize":
                        score = (pmax - val) / (pmax - pmin)
                    else:
                        score = (val - pmin) / (pmax - pmin)

                    score = max(0.0, min(1.0, score))

        utility += weight * score

    return utility
