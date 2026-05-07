"""Profile HuggingFace ``datasets.Dataset`` objects for tabular/constraint-benchmark use.

Use :func:`profile_dataset` for the profile dict, :func:`print_profile_report` for a full
terminal report, and :func:`print_retained_columns_report` for the retained-columns section
only. Use :func:`analyze_feature` for a single column.

Per-column ``details`` include, where applicable:

* **String (short)**: top ``TOP_K_VALUES`` value frequencies (value, count, share) for
  keep/trim decisions.
* **List/tuple cells** (e.g. tags, genres): rows are flattened like ``extend`` on list
  cells; per-row list-length stats, total elements, distinct labels, and top element
  frequencies.
* **Numeric**: top value frequencies (useful for discrete scores / ties); plus
  min/mean/std/percentiles, CV, IQR outliers, distinct counts.
* **Type-specific extras**: entropy, concentration, suggested filter ops, and a one-line
  ``column_summary`` on each column dict.

Requires: ``datasets``, ``numpy``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datasets import Dataset
    from datasets import IterableDataset

# How many most frequent values to show for keep/trim decisions
TOP_K_VALUES = 10

__all__ = [
    "analyze_feature",
    "profile_dataset",
    "profile_dataset_from_path",
    "profile_report_sections",
    "retained_columns",
    "format_retained_columns_text",
    "print_retained_columns_report",
    "print_profile_report",
    "TOP_K_VALUES",
]


def _as_list(x: Any) -> list:
    if hasattr(x, "to_pylist"):
        return x.to_pylist()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
        return list(x)
    return [x]


def _ensure_mappable_dataset(ds: Any) -> "Dataset":
    from datasets import Dataset, IterableDataset

    if isinstance(ds, IterableDataset):
        raise TypeError(
            "IterableDataset is not supported. Load with streaming=False or use "
            "Dataset.from_generator with a cap, or .take(n) and convert to a map-style Dataset."
        )
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(ds).__name__}")
    return ds


def _feature_dtype_str(dataset: "Dataset", column: str) -> str:
    return str(dataset.features[column]) if column in dataset.features else "unknown"


def _is_probably_id(name: str, n_unique: int, n: int) -> bool:
    if n_unique < n * 0.99:
        return False
    name_l = name.lower()
    return bool(
        re.search(
            r"(^|_)(id|key|uuid|guid|sku|asin|isbn|code|barcode|ean|upc|gtin|item_number|product_code|catalog_number)($|_)",
            name_l,
        )
        or name_l in ("id", "pk", "row_id", "item_id", "product_id", "code", "barcode", "sku")
    )


def _numeric_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"note": "no finite values"}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _string_stats(raw: list[Any]) -> dict[str, Any]:
    strs: list[str] = []
    for v in raw:
        if v is None:
            strs.append("")
        else:
            strs.append(str(v))
    lengths = [len(s) for s in strs if s]
    n_empty = sum(1 for s in strs if not s.strip())
    return {
        "n_empty_or_missing_text": n_empty,
        "char_len_min": int(min(lengths)) if lengths else 0,
        "char_len_max": int(max(lengths)) if lengths else 0,
        "char_len_mean": float(np.mean(lengths)) if lengths else 0.0,
    }


def _char_length_quantiles(strs: list[str]) -> dict[str, float]:
    """Extra length distribution for string columns (on non-empty strings)."""
    lengths = [len(s) for s in strs if s.strip()]
    if not lengths:
        return {}
    a = np.asarray(lengths, dtype=np.float64)
    return {
        "char_len_p50": float(np.percentile(a, 50)),
        "char_len_p90": float(np.percentile(a, 90)),
        "char_len_p99": float(np.percentile(a, 99)),
    }


def _shannon_entropy_normalized(counts: list[int], *, n: int) -> float:
    """0..1 spread measure: 0 = one value dominates, 1 = uniform (given support size)."""
    if n <= 0 or not counts:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    k = len([c for c in counts if c > 0])
    if k <= 1:
        return 0.0
    h_max = math.log2(k)
    return float(h / h_max) if h_max > 0 else 0.0


def _concentration_from_counter(ctr: Counter, *, n: int, top_ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Share of mass in the top-m categories (frequencies, sorted desc)."""
    if n <= 0:
        return {}
    sorted_counts = [c for _, c in ctr.most_common()]
    out: dict[str, float] = {}
    for m in top_ks:
        if m > len(sorted_counts):
            m = len(sorted_counts)
        top_sum = sum(sorted_counts[:m])
        out[f"cumulative_share_top_{m}"] = round(top_sum / n, 6)
    if sorted_counts:
        out["largest_class_share"] = round(sorted_counts[0] / n, 6)
    return out


def _categorical_extras_str(cells: list[str]) -> dict[str, Any]:
    """
    Rich summary for string columns (short categorical or long): frequencies, spread, examples.
    `cells` is str() of each non-null cell (empty string if blank).
    """
    n = len(cells)
    ctr = Counter(cells) if cells else Counter()
    n_unique = len(ctr)
    n_nonempty = sum(1 for s in cells if s and s.strip())
    extras: dict[str, Any] = {
        "n_non_empty_strings": n_nonempty,
        "n_unique_string_values": n_unique,
        "singleton_value_count": sum(1 for c in ctr.values() if c == 1),
    }
    stripped = [s for s in cells if s and s.strip()]
    if stripped:
        extras["char_length_quantiles"] = _char_length_quantiles(stripped)
    ctr_lower = Counter(s.strip().lower() for s in cells if s and s.strip())
    if ctr_lower and n_unique > len(ctr_lower):
        extras["n_unique_if_lowercased"] = len(ctr_lower)
        extras["note_case_or_whitespace"] = (
            f"Lowercase+trim would reduce {n_unique} to {len(ctr_lower)} distinct values; "
            "consider normalization for filters."
        )
    if n:
        c_list = list(ctr.values())
        extras["value_entropy_normalized_0_1"] = _shannon_entropy_normalized(c_list, n=n)
        extras.update(_concentration_from_counter(ctr, n=n, top_ks=(1, 3, 5, 10)))
    examples: list[str] = []
    for val, _ in ctr.most_common(5):
        s = str(val) if not isinstance(val, str) else val
        if s not in examples:
            examples.append(s[:300] if len(s) > 300 else s)
        if len(examples) >= 5:
            break
    for k, c in ctr.items():
        if c == 1 and str(k).strip() and str(k)[:300] not in examples and len(examples) < 5:
            sk = str(k)[:300]
            examples.append(sk)
    extras["example_values"] = examples[:5]
    if n_unique == 1:
        extras["suggested_filter_ops"] = ["eq"]
    elif n_unique <= 20:
        extras["suggested_filter_ops"] = ["in_set", "eq"]
    elif n_unique <= 5000:
        extras["suggested_filter_ops"] = ["in_set", "eq", "ilike (after normalize)"]
    else:
        extras["suggested_filter_ops"] = ["high_cardinality: bucket or embed; avoid raw one-hot in tasks"]
    return extras


def _numeric_extras(nums: list[float]) -> dict[str, Any]:
    if not nums:
        return {}
    a = np.asarray(nums, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"note": "no finite values"}
    q25, q75 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
    iqr = q75 - q25
    mean = float(np.mean(a))
    std = float(np.std(a))
    cv = float(std / abs(mean)) if abs(mean) > 1e-12 else None
    # discrete-ish count
    rounded = [round(x, 8) for x in a.tolist()]
    n_distinct = len(set(rounded))
    out: dict[str, Any] = {
        "n_finite": int(a.size),
        "n_distinct_values_approx": n_distinct,
        "iqr": round(iqr, 8) if iqr is not None else 0.0,
        "coefficient_of_variation": round(cv, 6) if cv is not None else None,
        "share_zero": round(float(np.sum(a == 0) / a.size), 6),
        "n_negative": int(np.sum(a < 0)),
    }
    if iqr and iqr > 0:
        lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        n_out = int(np.sum((a < lo) | (a > hi)))
        out["outlier_count_tukey_iqr"] = n_out
        out["outlier_share"] = round(n_out / a.size, 6)
    # discrete concentration from value counts
    ctr = Counter(rounded)
    if len(ctr) <= 5000:  # avoid huge counters
        out["value_entropy_normalized_0_1"] = _shannon_entropy_normalized(
            list(ctr.values()), n=int(a.size)
        )
        out.update(_concentration_from_counter(ctr, n=int(a.size), top_ks=(1, 3, 5, 10)))
    return out


def _list_column_extras(flat: list[Any], non_null: list[Any]) -> dict[str, Any]:
    """Per-element and row-level list stats after flattening."""
    if not flat and not non_null:
        return {}
    # element python types
    type_ctr = Counter(type(e).__name__ for e in flat)
    extras: dict[str, Any] = {
        "element_types_distribution": dict(type_ctr.most_common(8)),
        "avg_elements_per_row_all_non_null": round(
            len(flat) / max(len(non_null), 1), 4
        ),
    }
    if not flat:
        return extras
    lab = [_hashable_label(x) for x in flat]
    ctr = Counter(lab)
    extras["n_singleton_element_labels"] = sum(1 for c in ctr.values() if c == 1)
    extras["element_entropy_normalized_0_1"] = _shannon_entropy_normalized(
        list(ctr.values()), n=len(flat)
    )
    extras.update(
        _concentration_from_counter(ctr, n=len(flat), top_ks=(1, 3, 5, 10))
    )
    for v in non_null:
        if isinstance(v, (list, tuple)) and v:
            try:
                extras["example_row_list"] = [ _hashable_label(x) for x in v[:20] ]
            except Exception:
                extras["example_row_list"] = [str(x) for x in v[:20]]
            break
    return extras


def _dict_column_scan(non_null: list[dict], *, max_keys: int = 40) -> dict[str, Any]:
    """Aggregate key frequency across dict rows (first N dicts in sample)."""
    key_ctr: Counter[str] = Counter()
    for d in non_null[:5_000]:
        if not isinstance(d, dict):
            continue
        for k in d:
            key_ctr[str(k)] += 1
    top_keys = key_ctr.most_common(max_keys)
    return {
        "n_dict_rows_scanned": min(len(non_null), 5_000),
        "n_distinct_top_level_keys": len(key_ctr),
        "most_common_top_level_keys": [{"key": k, "row_count": c} for k, c in top_keys],
    }


def _subsample_list(items: list[Any], max_n: int = 8000) -> list[Any]:
    """Evenly subsample when list is huge (entropy / counter cost)."""
    if len(items) <= max_n:
        return items
    step = max(1, len(items) // max_n)
    return items[::step][:max_n]


def _long_text_extras(strs: list[str]) -> dict[str, Any]:
    """Heuristics for description-like columns."""
    words_per_row: list[int] = []
    for s in strs:
        if not s or not s.strip():
            continue
        words_per_row.append(len(s.split()))
    if not words_per_row:
        return {"note": "all empty for word stats"}
    a = np.asarray(words_per_row, dtype=np.float64)
    n_dup = len(strs) - len(set(strs))
    return {
        "word_count_per_row_p50": float(np.percentile(a, 50)),
        "word_count_per_row_p90": float(np.percentile(a, 90)),
        "word_count_max": int(np.max(a)),
        "approx_exact_duplicate_rows": n_dup,
        "preview_truncated_200_chars": [
            s[:200] + "…" if len(s) > 200 else s
            for s in strs[:3]
        ],
    }


def _build_column_summary_line(
    kind: str,
    details: dict[str, Any],
    *,
    n_non_null: int,
    n_unique: int,
    null_rate: float,
) -> str:
    """One compact paragraph describing the column (stored as ``details['column_summary']``)."""
    if kind == "all_null":
        return "All values null in this sample — consider dropping column or fixing parsing."

    parts: list[str] = [f"Column type: {kind}."]
    parts.append(
        f"Non-null {n_non_null}, null {null_rate:.1%}, unique row-level values {n_unique}."
    )
    if kind in ("string_categorical", "string_long_text") and details.get("value_entropy_normalized_0_1") is not None:
        parts.append(
            f"String value spread (entropy 0-1)={details['value_entropy_normalized_0_1']:.2f} "
            f"(1=even). Top-1 class share {details.get('largest_class_share', 'n/a')}."
        )
    if kind == "list_of_values" and details.get("element_entropy_normalized_0_1") is not None:
        ne = details.get("n_elements_total_flattened")
        parts.append(
            f"After flattening lists: {ne} element instances; element-entropy(0-1)={details['element_entropy_normalized_0_1']:.2f}."
        )
    if kind in ("int_numeric", "float_numeric", "datetime") and details:
        if details.get("coefficient_of_variation") is not None:
            parts.append(
                f"CV={details['coefficient_of_variation']:.4g}; distinct ≈{details.get('n_distinct_values_approx', '?')}; "
                f"outlier share (Tukey)={details.get('outlier_share', 'n/a')}."
            )
        else:
            parts.append(
                f"Distinct ≈{details.get('n_distinct_values_approx', '?')}; use range/threshold filters."
            )
    if kind == "bool":
        tr = details.get("true_rate")
        if tr is not None:
            bal = "balanced" if 0.25 < tr < 0.75 else "skewed"
            parts.append(f"Boolean true_rate={tr:.1%} ({bal}).")
    if kind == "string_long_text" and details.get("word_count_per_row_p50") is not None:
        parts.append(
            f"~{details['word_count_per_row_p50']:.0f} words/row p50, max {details.get('word_count_max', '?')} words; use for text constraints not enum filters."
        )
    if kind == "complex_nested":
        parts.append(
            f"JSON/dict: {details.get('n_distinct_top_level_keys', '?')} key names seen across rows; flatten before eq/in filters."
        )
    if details.get("note_case_or_whitespace"):
        parts.append(details["note_case_or_whitespace"])
    if kind == "string_mixed":
        parts.append("Could not read as a single type; treat as string or clean before filtering.")
    if kind == "mixed_or_other":
        parts.append("Mixed python types; cast or split column before benchmark use.")
    return " ".join(parts)


def _extend_details_from_type_blob(details: dict[str, Any], blob: dict[str, Any]) -> None:
    """Merge type-specific metrics into ``details`` (no nested report dict)."""
    for k, v in blob.items():
        if k == "kind":
            continue
        details[k] = v


def _structure_hint(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "null_only_sample"}
    if isinstance(value, dict):
        keys = list(value.keys())[:30]
        return {
            "kind": "object/dict",
            "sample_top_level_keys": keys,
            "n_keys_sample": len(value),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "sample_len": len(value),
            "first_element_type": type(value[0]).__name__ if value else "empty",
        }
    return {"kind": type(value).__name__}


def _hashable_label(v: Any) -> str:
    """One label per distinct value for counting (JSON-safe strings)."""
    if v is None:
        return "<null>"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "<nonfinite>"
        return str(int(v)) if float(v) == int(v) and abs(v) < 1e12 else str(v)
    if isinstance(v, (dict, list, tuple)):
        try:
            s = json.dumps(v, sort_keys=True, default=str)
        except (TypeError, ValueError):
            s = repr(v)
    else:
        s = str(v)
    if len(s) > 200:
        return s[:197] + "…"
    return s


def _flatten_multivalue_rows(non_null: list[Any]) -> list[Any]:
    """Flatten per-row list/tuple cells into a single stream of element values (user's list-column logic, generalized)."""
    out: list[Any] = []
    for v in non_null:
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            for e in v:
                if e is None:
                    continue
                if isinstance(e, float) and (math.isnan(e) or math.isinf(e)):
                    continue
                out.append(e)
        else:
            out.append(v)
    return out


def _list_length_stats(non_null: list[Any]) -> dict[str, Any]:
    """Per-row list lengths (only rows that are list/tuple)."""
    lens: list[int] = []
    for v in non_null:
        if v is not None and isinstance(v, (list, tuple)):
            lens.append(len(v))
    if not lens:
        return {"n_rows_as_list": 0, "list_len_per_row": None}
    a = np.asarray(lens, dtype=np.float64)
    return {
        "n_rows_as_list": len(lens),
        "list_len_per_row": {
            "min": int(np.min(a)),
            "max": int(np.max(a)),
            "mean": float(np.mean(a)),
            "p50": float(np.median(a)),
        },
    }


def _top_k_frequencies(
    values: list[Any],
    *,
    k: int = TOP_K_VALUES,
) -> list[dict[str, Any]]:
    """
    Return top *k* values by count with share of *values* in the list (after flattening for multi-value).
    """
    if not values:
        return []
    keys = [_hashable_label(x) for x in values]
    n = len(keys)
    ctr = Counter(keys)
    top = ctr.most_common(k)
    return [
        {"value": val, "count": cnt, "share": round(cnt / n, 6)}
        for val, cnt in top
    ]


def _full_value_frequencies(
    values: list[Any],
) -> list[dict[str, Any]]:
    """Return ALL values by count with share, sorted by descending frequency.

    Used by the config generator for adaptive sampling — the top_10 is kept
    for the triage LLM context window, but config generation needs the full
    distribution to pick representative sampling values.
    """
    if not values:
        return []
    keys = [_hashable_label(x) for x in values]
    n = len(keys)
    ctr = Counter(keys)
    return [
        {"value": val, "count": cnt, "share": round(cnt / n, 6)}
        for val, cnt in ctr.most_common()
    ]


def _is_temporal_scalar(x: Any) -> bool:
    """True for scalar date/time values we can map to POSIX seconds."""
    if x is None:
        return True
    if isinstance(x, dt.datetime):
        return True
    if isinstance(x, dt.date):
        return True
    if isinstance(x, np.datetime64):
        return True
    # pandas.Timestamp from pyarrow / parquet
    if type(x).__name__ == "Timestamp" and getattr(type(x), "__module__", "").startswith(
        "pandas",
    ):
        return True
    return False


def _scalar_to_posix_seconds(x: Any) -> float:
    """Convert a temporal scalar to UTC POSIX seconds (float)."""
    if x is None:
        raise TypeError("unexpected None")
    if isinstance(x, dt.datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        else:
            x = x.astimezone(dt.timezone.utc)
        return x.timestamp()
    if isinstance(x, dt.date):
        return dt.datetime.combine(x, dt.time.min, tzinfo=dt.timezone.utc).timestamp()
    if isinstance(x, np.datetime64):
        base = x.astype("datetime64[ns]")
        epoch = np.datetime64(0, "ns")
        return float((base - epoch) / np.timedelta64(1, "s"))
    if type(x).__name__ == "Timestamp" and getattr(type(x), "__module__", "").startswith(
        "pandas",
    ):
        return float(x.timestamp())
    raise TypeError(f"not a temporal scalar: {type(x)!r}")


def _is_list_column(non_null: list[Any]) -> bool:
    """True if this column is mostly per-row list/tuple (multi-value / tags), not a dict blob."""
    if not non_null:
        return False
    n = len(non_null)
    n_list = sum(1 for x in non_null if isinstance(x, (list, tuple)))
    return n_list >= n * 0.5


@dataclass
class ColumnSummary:
    name: str
    feature_dtype: str
    n_analyzed: int
    n_non_null: int
    null_rate: float
    n_unique: int
    uniqueness_ratio: float
    inferred_kind: str
    details: dict[str, Any] = field(default_factory=dict)
    quality_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_dtype": self.feature_dtype,
            "n_analyzed": self.n_analyzed,
            "n_non_null": self.n_non_null,
            "null_rate": round(self.null_rate, 4),
            "n_unique": self.n_unique,
            "uniqueness_ratio": round(self.uniqueness_ratio, 4),
            "inferred_kind": self.inferred_kind,
            "column_summary": self.details.get("column_summary"),
            "details": self.details,
            "quality_tags": self.quality_tags,
        }


def _unique_count(non_null: list[Any]) -> int:
    def key(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (dict, list)):
            try:
                return json.dumps(x, sort_keys=True, default=str)
            except (TypeError, ValueError):
                return repr(x)
        return x

    return len({key(x) for x in non_null})


def _score_column(summary: ColumnSummary) -> list[str]:
    tags: list[str] = []
    n = max(summary.n_analyzed, 1)
    null_r = summary.null_rate
    u = summary.n_unique

    if summary.inferred_kind in ("int_numeric", "float_numeric", "datetime"):
        tags.append("filterable_numeric")
        if null_r > 0.4:
            tags.append("sparse")
        d = summary.details
        if d.get("std") is not None and d.get("std", 0) > 0:
            tags.append("has_variance")
        else:
            tags.append("low_or_no_variance")
        if 3 <= u <= 5000:
            tags.append("good_cardinality_for_buckets")
        elif u <= 2:
            tags.append("binary_or_trivial")
    elif summary.inferred_kind == "string_categorical":
        if null_r < 0.2 and 2 <= u <= 10000:
            tags.append("filterable_categorical")
        if u == 1:
            tags.append("constant_column")
        if u > n * 0.5:
            tags.append("high_uniqueness_like_free_text")
    elif summary.inferred_kind == "list_of_values":
        tags.append("multi_value_per_row")
        el = (summary.details or {}).get("n_distinct_element_labels")
        if el is not None and 2 <= el <= 10000:
            tags.append("filterable_after_flattening_or_membership")
        if el is not None and el == 1:
            tags.append("single_repeated_tag_only")
    elif summary.inferred_kind == "string_long_text":
        tags.append("long_text_may_support_description_constraints")
    elif summary.inferred_kind == "bool":
        tags.append("filterable_boolean")
    else:
        tags.append("limited_tabular_filter_use")

    if _is_probably_id(summary.name, u, n):
        tags.append("candidate_identifier_column")

    # Name-based noise detection
    name_l = summary.name.lower()
    if any(kw in name_l for kw in ("_url", "url_", "image_", "_image", "img_",
                                    "_img", "photo", "thumbnail", "icon_",
                                    "_icon", "logo_", "_logo")):
        tags.append("noise_media_reference")
    elif re.match(
        r".*(creator|editors?|checkers?|correctors?|informers?|photographers?).*",
        name_l,
    ):
        tags.append("noise_audit_trail")
    elif any(kw in name_l for kw in ("completeness", "data_quality", "states_tags",
                                      "ecoscore_data", "nutriments_estimated")):
        tags.append("noise_system_metadata")
    elif re.match(r"^(created|updated|modified|last_edit|entry_dates?)", name_l):
        tags.append("noise_timestamp")

    # Near-empty list columns (< 100 elements across entire sample)
    if summary.inferred_kind == "list_of_values":
        total_el = (summary.details or {}).get("n_elements_total_flattened", 0)
        if total_el < 100 and summary.n_analyzed >= 1000:
            tags.append("noise_near_empty_list")

    return tags


# ---------------------------------------------------------------------------
# Noise detection (cross-column)
# ---------------------------------------------------------------------------

# Minimum length for a name to count as a meaningful prefix (avoids tiny stems like ``id``).
_PREFIX_SIBLING_MIN_STEM_LEN = 3


def _prefix_sibling_column_names(name: str, all_names: set[str]) -> list[str]:
    """For ``name``, scan every other column for prefix-based siblings.

    If **current** is a long-enough prefix of **other** (and ``other`` is longer), they
    relate; the same loop also catches **other** as a long-enough prefix of **current**
    so hints are symmetric (e.g. ``labels`` ↔ ``labels_tags``).
    """
    sibs: list[str] = []
    min_len = _PREFIX_SIBLING_MIN_STEM_LEN
    for other in all_names:
        if other == name:
            continue
        if len(name) >= min_len and len(other) > len(name) and other.startswith(name):
            sibs.append(other)
        elif len(other) >= min_len and len(name) > len(other) and name.startswith(other):
            sibs.append(other)
    return sorted(set(sibs))


_NOISE_TAG_TO_REASON: dict[str, str] = {
    "noise_media_reference": "media reference (URL/image)",
    "noise_audit_trail": "audit trail field",
    "noise_system_metadata": "system metadata",
    "noise_timestamp": "timestamp / date metadata",
    "noise_near_empty_list": "near-empty list column",
}


def _attach_possible_sibling_columns(summaries: list[ColumnSummary]) -> None:
    """Mutate each summary's ``details`` with non-blocking sibling hints for triage prompts.

    Siblings share a **string-prefix** relationship (one column name is a prefix of another,
    with minimum stem length). The LLM chooses which to keep for filtering.
    """
    all_names = {s.name for s in summaries}
    for s in summaries:
        sibs = _prefix_sibling_column_names(s.name, all_names)
        if sibs:
            s.details["possible_sibling_columns"] = sibs


def _detect_noise_columns(
    summaries: list[ColumnSummary],
) -> dict[str, str]:
    """Identify columns that are clearly noise for benchmark purposes.

    Returns ``{column_name: human-readable reason}``.
    """
    noise: dict[str, str] = {}

    for s in summaries:
        # Already detected by quality tags
        for tag, reason in _NOISE_TAG_TO_REASON.items():
            if tag in s.quality_tags:
                if tag == "noise_near_empty_list":
                    total = (s.details or {}).get("n_elements_total_flattened", 0)
                    noise[s.name] = f"near-empty list ({total} elements in {s.n_analyzed:,} rows)"
                else:
                    noise[s.name] = reason
                break
        if s.name in noise:
            continue

        if s.inferred_kind == "all_null":
            noise[s.name] = "all values null"
            continue
        if s.n_unique <= 1 and s.n_analyzed > 0:
            noise[s.name] = "constant column (1 unique value)"
            continue

    return noise


def analyze_feature(
    dataset: "Dataset",
    column: str,
    *,
    max_rows: int = 200_000,
) -> dict[str, Any]:
    """Per-column analysis; returns JSON-serializable dict with `summary` key."""
    _ensure_mappable_dataset(dataset)
    if column not in dataset.column_names:
        raise KeyError(f"Unknown column: {column!r}. Available: {dataset.column_names}")

    n_total = len(dataset)
    n_take = min(n_total, max(1, int(max_rows)))
    sub = dataset.select(range(n_take))
    raw = _as_list(sub[column])
    feature_dtype = _feature_dtype_str(dataset, column)

    n = len(raw)
    non_null = [x for x in raw if x is not None]
    n_non_null = len(non_null)
    null_rate = 1.0 - (n_non_null / n) if n else 1.0

    n_unique = _unique_count(non_null)
    uniqueness_ratio = n_unique / n_non_null if n_non_null else 0.0

    details: dict[str, Any] = {}
    inferred_kind = "unknown"

    if not non_null:
        inferred_kind = "all_null"
    else:
        sample = next((x for x in raw if x is not None), None)
        # Struct / object columns (e.g. nested JSON) — not the same as list-of-tags
        if sample is not None and isinstance(sample, dict):
            inferred_kind = "complex_nested"
            details["structure"] = _structure_hint(sample)
            dict_rows = [x for x in non_null if isinstance(x, dict)]
            _extend_details_from_type_blob(
                details, {**_dict_column_scan(dict_rows), "kind": "complex_nested"}
            )
        # Per-row list/tuple: genres, tags, ports — aggregate element frequencies
        elif _is_list_column(non_null):
            inferred_kind = "list_of_values"
            flat = _flatten_multivalue_rows(non_null)
            details["list_row_stats"] = _list_length_stats(non_null)
            details["n_elements_total_flattened"] = len(flat)
            if flat:
                labels = [_hashable_label(x) for x in flat]
                details["n_distinct_element_labels"] = len(set(labels))
                details[f"top_{TOP_K_VALUES}_element_frequencies"] = _top_k_frequencies(
                    flat, k=TOP_K_VALUES
                )
                details["element_frequencies_full"] = _full_value_frequencies(flat)
            else:
                details["n_distinct_element_labels"] = 0
            _extend_details_from_type_blob(
                details,
                {
                    **_list_column_extras(flat, non_null),
                    "n_elements_total_flattened": len(flat),
                    "kind": "list_of_values",
                },
            )
        elif all(
            x is None or isinstance(x, (bool, np.bool_)) for x in non_null
        ) or (all(x is None or x in (0, 1) for x in non_null) and not any(
            isinstance(x, (int, float)) and x not in (0, 1) for x in non_null
        )):
            inferred_kind = "bool"
            trues = sum(1 for x in non_null if bool(x))
            tr = trues / len(non_null)
            details["true_rate"] = tr
            _extend_details_from_type_blob(
                details,
                {
                    "true_rate": tr,
                    "false_rate": round(1.0 - tr, 6),
                    "class_balance": "balanced" if 0.25 < tr < 0.75 else "skewed",
                    "suggested_filter_ops": ["eq"],
                    "kind": "bool",
                },
            )
        elif all(x is None or _is_temporal_scalar(x) for x in non_null):
            epoch_vals: list[float] = []
            temporal_ok = True
            for x in non_null:
                if x is None:
                    continue
                try:
                    epoch_vals.append(_scalar_to_posix_seconds(x))
                except (TypeError, ValueError, OSError, OverflowError):
                    temporal_ok = False
                    break
            if temporal_ok and epoch_vals:
                inferred_kind = "datetime"
                details.update(_numeric_stats(epoch_vals))
                details[f"top_{TOP_K_VALUES}_value_frequencies"] = _top_k_frequencies(
                    epoch_vals, k=TOP_K_VALUES
                )
                _extend_details_from_type_blob(
                    details,
                    {
                        **_numeric_extras(epoch_vals),
                        "suggested_filter_ops": [">=", "<="],
                        "kind": "datetime",
                        "note": (
                            "Date/time column; filters compare as datetimes. "
                            "Sampling uses ISO calendar dates (YYYY-MM-DD), UTC-normalized."
                        ),
                    },
                )
            else:
                inferred_kind = "mixed_or_other"
                details["python_types_seen"] = list(
                    {type(x).__name__ for x in non_null[:500]}
                )[:20]
                _extend_details_from_type_blob(
                    details,
                    {
                        "observed_value_types": details["python_types_seen"],
                        "kind": "mixed_or_other",
                        "note": "Cast to one type or split into multiple columns for filters.",
                    },
                )
        elif all(
            x is None or (isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_)))
            for x in non_null
        ):
            try:
                nums = []
                for x in non_null:
                    v = float(x)
                    if math.isfinite(v):
                        nums.append(v)
                inferred_kind = "int_numeric" if nums and all(v == int(v) for v in nums) else "float_numeric"
                if nums:
                    details.update(_numeric_stats(nums))
                    details[f"top_{TOP_K_VALUES}_value_frequencies"] = _top_k_frequencies(
                        nums, k=TOP_K_VALUES
                    )
                    _extend_details_from_type_blob(
                        details,
                        {
                            **_numeric_extras(nums),
                            "suggested_filter_ops": [">=", "<=", "eq", "in (if few distinct)"],
                            "kind": inferred_kind,
                        },
                    )
                else:
                    _extend_details_from_type_blob(
                        details,
                        {
                            "note": "no finite numeric values in sample (NaN/inf or empty)",
                            "kind": inferred_kind,
                        },
                    )
            except (TypeError, ValueError):
                inferred_kind = "string_mixed"
                _extend_details_from_type_blob(
                    details,
                    {
                        "kind": "string_mixed",
                        "note": "coercion to number failed; treat as string or clean.",
                    },
                )
        elif all(x is None or isinstance(x, str) for x in non_null):
            str_st = _string_stats(non_null)
            details["string_stats"] = str_st
            str_cells = [str(x) for x in non_null]
            if str_st.get("char_len_mean", 0) > 120 or str_st.get("char_len_max", 0) > 500:
                inferred_kind = "string_long_text"
                if str_st.get("char_len_max", 0) < 2000:
                    details[f"top_{TOP_K_VALUES}_value_frequencies"] = _top_k_frequencies(
                        non_null, k=TOP_K_VALUES
                    )
                    details["value_frequencies_full"] = _full_value_frequencies(non_null)
                else:
                    details["value_frequency_note"] = (
                        "very_long_strings; exact-value frequencies may be nearly unique; "
                        "use token/hash stats below or subsample-based spread."
                    )
                sub = _subsample_list(str_cells, 8000)
                cat_part = _categorical_extras_str(sub)
                cat_part["categorical_subsample"] = f"{len(sub)} of {len(str_cells)} rows for spread metrics"
                _extend_details_from_type_blob(
                    details,
                    {**_long_text_extras(str_cells), **cat_part, "kind": "string_long_text"},
                )
            else:
                inferred_kind = "string_categorical"
                details[f"top_{TOP_K_VALUES}_value_frequencies"] = _top_k_frequencies(
                    non_null, k=TOP_K_VALUES
                )
                details["value_frequencies_full"] = _full_value_frequencies(non_null)
                _extend_details_from_type_blob(
                    details,
                    {**_categorical_extras_str(str_cells), "kind": "string_categorical"},
                )
        else:
            inferred_kind = "mixed_or_other"
            details["python_types_seen"] = list(
                {type(x).__name__ for x in non_null[:500]}
            )[:20]
            _extend_details_from_type_blob(
                details,
                {
                    "observed_value_types": details["python_types_seen"],
                    "kind": "mixed_or_other",
                    "note": "Cast to one type or split into multiple columns for filters.",
                },
            )

    details["column_summary"] = _build_column_summary_line(
        inferred_kind, details, n_non_null=n_non_null, n_unique=n_unique, null_rate=null_rate
    )

    summary = ColumnSummary(
        name=column,
        feature_dtype=feature_dtype,
        n_analyzed=n,
        n_non_null=n_non_null,
        null_rate=null_rate,
        n_unique=n_unique,
        uniqueness_ratio=uniqueness_ratio,
        inferred_kind=inferred_kind,
        details=details,
    )
    summary.quality_tags = _score_column(summary)

    out: dict[str, Any] = {"summary": summary.to_dict()}
    if n_total > n:
        out["note"] = f"Analyzed first {n} of {n_total} rows; pass max_rows to change."
    return out


def _overall_assessment(
    n_total: int,
    n_rows_analyzed: int,
    by_col: list[ColumnSummary],
    noise_columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    n_cols = len(by_col)
    if n_cols == 0:
        return {
            "verdict": "poor",
            "score_1_10": 1,
            "paragraph": "The dataset has no columns.",
        }

    noise_set = set(noise_columns or {})
    retained = [s for s in by_col if s.name not in noise_set]

    def has_tag(s: ColumnSummary, t: str) -> bool:
        return t in s.quality_tags

    n_numeric = sum(
        1
        for s in retained
        if s.inferred_kind in ("int_numeric", "float_numeric", "datetime")
        and "has_variance" in s.quality_tags
    )
    n_cat = sum(1 for s in retained if has_tag(s, "filterable_categorical"))
    n_bool = sum(1 for s in retained if s.inferred_kind == "bool" or has_tag(s, "filterable_boolean"))
    n_set = sum(
        1
        for s in retained
        if s.inferred_kind == "list_of_values" and has_tag(s, "filterable_after_flattening_or_membership")
    )
    n_text = sum(1 for s in retained if s.inferred_kind == "string_long_text")
    n_sparse = sum(1 for s in retained if s.null_rate > 0.5)
    well_filled = sum(1 for s in retained if s.null_rate <= 0.2)
    n_noise = len(noise_set)

    filter_like = n_numeric + n_cat + n_bool + n_set
    score = 5.0
    if n_total < 1_000:
        score -= 1.0
    elif n_total >= 10_000:
        score += 0.5
    if filter_like >= 8:
        score += 1.5
    elif filter_like >= 4:
        score += 0.5
    if n_numeric >= 2:
        score += 0.5
    if n_text >= 1:
        score += 0.3
    if n_set >= 1:
        score += 0.3
    if n_sparse > len(retained) * 0.5:
        score -= 1.0
    if well_filled < max(2, len(retained) // 3):
        score -= 0.5
    score = max(1, min(10, int(round(score))))

    if score >= 8:
        verdict = "strong"
    elif score >= 6:
        verdict = "good"
    elif score >= 4:
        verdict = "fair"
    else:
        verdict = "weak"

    paragraph = (
        f"Verdict: {verdict} (heuristic score {score}/10) for *tabular constraint / "
        f"recommendation-benchmark* use. {n_total:,} rows, {n_cols} columns "
        f"({n_noise} auto-dropped as noise, {len(retained)} retained). "
        f"Filterable: {n_numeric} numeric, {n_cat} categorical, {n_bool} boolean, "
        f"{n_set} set-valued. {n_text} text field(s) for embeddings / desc-constraints. "
        f"{well_filled} columns <=20% null, {n_sparse} columns >50% null."
    )

    return {
        "verdict": verdict,
        "score_1_10": score,
        "paragraph": paragraph,
        "stats": {
            "n_rows_total": n_total,
            "n_rows_profiled": n_rows_analyzed,
            "n_columns_total": n_cols,
            "n_columns_retained": len(retained),
            "n_columns_noise": n_noise,
            "n_filterable_numeric": n_numeric,
            "n_filterable_categorical": n_cat,
            "n_filterable_boolean": n_bool,
            "n_filterable_set_valued": n_set,
            "n_long_text": n_text,
            "n_sparse_columns": n_sparse,
            "n_well_filled": well_filled,
        },
    }


def profile_dataset(
    dataset: Any,
    *,
    max_rows: int = 200_000,
) -> dict[str, Any]:
    """Profile a HuggingFace :class:`datasets.Dataset` or a pandas DataFrame."""

    import pandas as pd
    from datasets import Dataset

    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_pandas(dataset, preserve_index=False)
    _ensure_mappable_dataset(dataset)
    n_total = len(dataset)
    n_rows = min(n_total, max(1, int(max_rows)))

    summaries: list[ColumnSummary] = []
    for col in sorted(dataset.column_names):
        d = analyze_feature(dataset, col, max_rows=n_rows)
        sdict = d["summary"]
        summaries.append(
            ColumnSummary(
                name=sdict["name"],
                feature_dtype=sdict["feature_dtype"],
                n_analyzed=sdict["n_analyzed"],
                n_non_null=sdict["n_non_null"],
                null_rate=sdict["null_rate"],
                n_unique=sdict["n_unique"],
                uniqueness_ratio=sdict["uniqueness_ratio"],
                inferred_kind=sdict["inferred_kind"],
                details=sdict.get("details", {}),
                quality_tags=sdict.get("quality_tags", []),
            )
        )

    _attach_possible_sibling_columns(summaries)
    noise = _detect_noise_columns(summaries)
    assessment = _overall_assessment(n_total, n_rows, summaries, noise)
    return {
        "n_rows_total": n_total,
        "n_rows_profiled": n_rows,
        "assessment": assessment,
        "noise_columns": noise,
        "columns": [s.to_dict() for s in sorted(summaries, key=lambda x: x.name)],
    }


def _fmt_top_values(freqs: list[dict[str, Any]], *, limit: int = 5) -> str:
    """Format top-k value frequencies as a compact one-liner."""
    parts: list[str] = []
    for row in freqs[:limit]:
        val = row.get("value", "?")
        share = row.get("share", 0)
        parts.append(f"{val} ({share:.1%})")
    return ", ".join(parts)


def retained_columns(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Columns not in ``profile['noise_columns']``, same order as ``profile['columns']``."""
    noise: dict[str, str] = profile.get("noise_columns", {})
    return [c for c in profile["columns"] if c["name"] not in noise]


def format_retained_columns_text(retained: list[dict[str, Any]]) -> str:
    """Human-readable block for *retained* columns only (used by the print helpers)."""
    lines: list[str] = []
    for c in retained:
        kind = c["inferred_kind"]
        det = c.get("details") or {}
        tags = c.get("quality_tags", [])

        fill_pct = (1.0 - c["null_rate"]) * 100
        lines.append("")
        lines.append(f"  {c['name']}  [{kind}]")

        base = f"    {fill_pct:.0f}% filled | {c['n_unique']:,} distinct"

        if kind in ("int_numeric", "float_numeric", "datetime"):
            lo = det.get("min")
            hi = det.get("max")
            mean = det.get("mean")
            p50 = det.get("p50")
            if lo is not None:
                base += f" | range [{lo}, {hi}]  mean {mean:.4g}  p50 {p50:.4g}"
            topk = det.get(f"top_{TOP_K_VALUES}_value_frequencies")
            lines.append(base)
            if topk:
                lines.append(f"    Top: {_fmt_top_values(topk)}")

        elif kind == "string_categorical":
            entropy = det.get("value_entropy_normalized_0_1")
            if entropy is not None:
                base += f" | entropy {entropy:.2f}"
            lines.append(base)
            topk = det.get(f"top_{TOP_K_VALUES}_value_frequencies")
            if topk:
                lines.append(f"    Top: {_fmt_top_values(topk)}")
            if det.get("note_case_or_whitespace"):
                lines.append(f"    Note: {det['note_case_or_whitespace']}")

        elif kind == "list_of_values":
            n_labels = det.get("n_distinct_element_labels", "?")
            lrs = det.get("list_row_stats") or {}
            llen = lrs.get("list_len_per_row")
            avg_per_row = f"  avg {llen['mean']:.1f}/row" if llen else ""
            base += f" | {n_labels} distinct labels{avg_per_row}"
            lines.append(base)
            tflat = det.get(f"top_{TOP_K_VALUES}_element_frequencies")
            if tflat:
                lines.append(f"    Top labels: {_fmt_top_values(tflat)}")

        elif kind == "bool":
            tr = det.get("true_rate")
            if tr is not None:
                bal = "balanced" if 0.25 < tr < 0.75 else "skewed"
                base += f" | true_rate {tr:.1%} ({bal})"
            lines.append(base)

        elif kind == "string_long_text":
            ss = det.get("string_stats", {})
            wc = det.get("word_count_per_row_p50")
            if wc is not None:
                base += f" | ~{wc:.0f} words/row (p50)"
            elif ss.get("char_len_mean"):
                base += f" | ~{ss['char_len_mean']:.0f} chars/row avg"
            lines.append(base)

        elif kind == "complex_nested":
            n_keys = det.get("n_distinct_top_level_keys", "?")
            base += f" | {n_keys} top-level keys"
            lines.append(base)

        else:
            lines.append(base)

        sibs = det.get("possible_sibling_columns")
        if sibs:
            lines.append(
                f"    Related columns (one name is a prefix of another — keep the best for "
                f"filtering, not necessarily all): {', '.join(sibs)}",
            )

        display_tags = [t for t in tags if not t.startswith("noise_")]
        if display_tags:
            lines.append(f"    tags: {', '.join(display_tags)}")

    return "\n".join(lines)


def profile_report_sections(profile: dict[str, Any]) -> dict[str, Any]:
    """Split a profile dict into pieces used by :func:`print_profile_report`."""
    noise: dict[str, str] = profile.get("noise_columns", {})
    retained = retained_columns(profile)
    return {
        "assessment": profile["assessment"],
        "noise_columns": noise,
        "retained": retained,
        "retained_columns_text": format_retained_columns_text(retained),
        "n_rows_total": profile["n_rows_total"],
        "n_rows_profiled": profile["n_rows_profiled"],
    }


def print_retained_columns_report(profile: dict[str, Any]) -> None:
    """Print only the retained-columns section (same body as inside :func:`print_profile_report`)."""
    retained = retained_columns(profile)
    print("-" * 72)
    print(f"RETAINED COLUMNS ({len(retained)})")
    print("-" * 72)
    print(format_retained_columns_text(retained))


def print_profile_report(
    dataset_or_profile: "Dataset | dict[str, Any]",
    *,
    max_rows: int = 200_000,
) -> None:
    """Print a full profile report. Pass a profile dict to avoid recomputing statistics."""
    if isinstance(dataset_or_profile, dict):
        data = dataset_or_profile
    else:
        data = profile_dataset(dataset_or_profile, max_rows=max_rows)

    a = data["assessment"]
    noise: dict[str, str] = data.get("noise_columns", {})

    print("=" * 72)
    print("DATASET PROFILE (constraint / recommendation-benchmark heuristics)")
    print("=" * 72)
    print()
    print(a["paragraph"])
    print()
    print("Key stats:", json.dumps(a.get("stats", {}), indent=2))

    print()
    print_retained_columns_report(data)

    if noise:
        print()
        print("-" * 72)
        print(f"AUTO-DROPPED COLUMNS ({len(noise)})")
        print("-" * 72)
        max_name = max(len(n) for n in noise) if noise else 0
        for name, reason in sorted(noise.items()):
            print(f"  {name:<{max_name}}  {reason}")

    print()
    print("=" * 72)
    if data["n_rows_total"] > data["n_rows_profiled"]:
        print(
            f"Note: only first {data['n_rows_profiled']:,} rows were scanned "
            f"(of {data['n_rows_total']:,})."
        )


def _read_dataframe_head(path: "Path", *, nrows: int) -> "Any":
    """Read at most *nrows* from a tabular file. Avoids full-file I/O and DataFrame size."""
    from pathlib import Path

    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    nrows = max(1, int(nrows))

    if suffix == ".csv":
        return pd.read_csv(p, nrows=nrows)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t", nrows=nrows)
    if suffix in (".jsonl", ".ndjson"):
        # Pandas nrows is fastest when available; line-by-line head avoids parsing the full file.
        try:
            return pd.read_json(p, lines=True, nrows=nrows)
        except (TypeError, ValueError):
            return _read_jsonl_head_lines(p, nrows)
    if suffix in (".parquet", ".pq"):
        return _read_parquet_head(p, nrows)
    try:
        return pd.read_json(p, lines=True, nrows=nrows)
    except (TypeError, ValueError, UnicodeDecodeError):
        pass
    try:
        return _read_jsonl_head_lines(p, nrows)
    except Exception:
        pass
    try:
        return _read_parquet_head(p, nrows)
    except Exception:
        pass
    return pd.read_parquet(p)  # last resort: may read full file


def _read_jsonl_head_lines(path: "Path", nrows: int) -> "Any":
    import json
    from pathlib import Path

    import pandas as pd

    p = Path(path)
    nrows = max(1, int(nrows))
    rows: list[dict] = []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= nrows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _read_parquet_head(path: "Path", nrows: int) -> "Any":
    """Load only the first *nrows* rows (iterates row groups; does not full-scan the file)."""
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    p = Path(path)
    nrows = max(1, int(nrows))
    pf = pq.ParquetFile(p)
    batches: list = []
    total = 0
    for batch in pf.iter_batches(batch_size=min(65_536, nrows)):
        if total + batch.num_rows > nrows:
            batch = batch.slice(0, nrows - total)
        batches.append(batch)
        total += batch.num_rows
        if total >= nrows:
            break
    if not batches:
        return pd.DataFrame()
    table = pa.Table.from_batches(batches)
    return table.slice(0, nrows).to_pandas()


def profile_dataset_from_path(
    path: str,
    *,
    max_rows: int = 200_000,
) -> dict[str, Any]:
    """Load a CSV / TSV / JSON / JSONL / Parquet table and return the same dict as
    :func:`profile_dataset`.

    Only the first *max_rows* file rows are read. Profiling only ever needs a bounded sample;
    reading and converting a multi-GB file to a HuggingFace :class:`datasets.Dataset` is what
    made the old "load everything" path very slow. The effective row count in the result is
    ``min(rows_on_disk, max_rows)``.
    """
    from pathlib import Path

    from datasets import Dataset

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    n_load = max(1, int(max_rows))
    df = _read_dataframe_head(p, nrows=n_load)

    ds = Dataset.from_pandas(df, preserve_index=False)
    return profile_dataset(ds, max_rows=max_rows)