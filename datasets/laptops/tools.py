"""Domain-specific tools for the laptop recommendation benchmark.

Provides four tools tailored to the 8,369-laptop catalog:
  1. filter_laptops        -- structured constraint filtering with sort/limit
  2. search_laptops        -- dual-mode search (keyword TF-IDF + semantic embeddings)
  3. get_laptop_details    -- full detail view for one laptop by product_id
  4. get_catalog_stats     -- summary statistics about the catalog

Entry point: ``build_domain_specific_tools()`` returns ``list[DomainToolSpec]``.
"""

from __future__ import annotations

import ast
import json as _json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.config import AttrType, ConstraintOp
from simulation.tools import DomainCatalogContext, DomainToolSpec, _fuzzy_resolve_constraints
from shared.filter import GenericFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEARCH_FIELDS = [
    "name",
    "brand",
    "processor",
    "gpu_type",
    "storage_type",
    "display_type",
    "os_version",
    "use_cases",
    "description",
]

_SORT_FIELDS = ["price", "ram_gb", "storage_gb", "weight_kg", "battery_hours",
                "screen_inches", "avg_rating", "num_reviews"]


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _normalize_text(text)) if len(t) >= 2]


def _fmt_price(val: Any) -> str:
    if _is_null(val):
        return "N/A"
    try:
        return f"${float(val):,.2f}"
    except Exception:
        return str(val)


def _fmt_int(val: Any) -> str:
    if _is_null(val):
        return "N/A"
    try:
        return f"{int(round(float(val))):,}"
    except Exception:
        return str(val)


def _fmt_float(val: Any, digits: int = 1) -> str:
    if _is_null(val):
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return str(val)


def _fmt_bool(val: Any) -> str:
    if _is_null(val):
        return "N/A"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    s = str(val).strip().lower()
    if s in {"true", "1", "yes"}:
        return "Yes"
    if s in {"false", "0", "no"}:
        return "No"
    return str(val)


def _safe_list(v: Any) -> list:
    if _is_null(v):
        return []
    if isinstance(v, (list, tuple)):
        return [x for x in v if not _is_null(x)]
    if isinstance(v, np.ndarray):
        return [x for x in v.tolist() if not _is_null(x)]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [x for x in parsed if not _is_null(x)]
            except Exception:
                pass
        return [s] if s else []
    return []


def _safe_num(val: Any) -> float | None:
    if _is_null(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_val(row: pd.Series, col: str) -> Any:
    try:
        val = row[col] if col in row.index else None
    except (KeyError, IndexError):
        return None
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _laptop_one_liner(row: pd.Series, pid: str) -> str:
    name = _safe_val(row, "name") or "Unknown"
    brand = _safe_val(row, "brand") or ""
    price = _fmt_price(_safe_val(row, "price"))
    ram = _safe_num(_safe_val(row, "ram_gb"))
    ram_s = f"{int(ram)}GB RAM" if ram is not None else ""
    storage = _safe_num(_safe_val(row, "storage_gb"))
    stype = _safe_val(row, "storage_type") or ""
    storage_s = f"{int(storage)}GB {stype}".strip() if storage is not None else ""
    screen = _safe_num(_safe_val(row, "screen_inches"))
    screen_s = f'{screen:.1f}"' if screen is not None else ""
    gpu = _safe_val(row, "gpu_type") or ""
    weight = _safe_num(_safe_val(row, "weight_kg"))
    weight_s = f"{weight:.2f}kg" if weight is not None else ""

    specs = "; ".join(s for s in [ram_s, storage_s, screen_s, gpu, weight_s] if s)
    return f"{name} ({pid}) — {brand}; {price}; {specs}"


def _extract_constraints(kwargs: dict) -> dict:
    mapping = {
        "price_max": "price_max_usd",
        "price_min": "price_min",
        "ram_min_gb": "ram_min_gb",
        "gpu_type": "gpu_type",
        "brand": "brand",
        "weight_max_kg": "weight_max_kg",
        "storage_min_gb": "storage_min_gb",
        "screen_min_inches": "screen_min_inches",
        "screen_max_inches": "screen_max_inches",
        "battery_min_hours": "battery_min_hours",
        "storage_type": "storage_type",
        "touchscreen": "touchscreen",
        "backlit_keyboard": "backlit_keyboard",
    }
    constraints = {}
    for param, cname in mapping.items():
        v = kwargs.get(param)
        if v is None:
            continue
        if param in ("touchscreen", "backlit_keyboard"):
            if isinstance(v, str):
                v = v.strip().lower() in {"true", "1", "yes"}
            else:
                v = bool(v)
        elif param in ("price_max", "price_min", "weight_max_kg",
                       "screen_min_inches", "screen_max_inches",
                       "battery_min_hours"):
            try:
                v = float(v)
            except Exception:
                continue
        elif param in ("ram_min_gb", "storage_min_gb"):
            try:
                v = int(v)
            except Exception:
                continue
        elif param in ("gpu_type", "brand", "storage_type"):
            v = str(v).strip()
        constraints[cname] = v
    return constraints


def _fuzzy_field_match(user_field: str, valid_fields: list[str]) -> str | None:
    user_lower = user_field.lower().strip().replace(" ", "_")
    for f in valid_fields:
        if user_lower == f.lower():
            return f

    aliases = {
        "manufacturer": "brand",
        "make": "brand",
        "company": "brand",
        "price": "price",
        "cost": "price",
        "memory": "ram_gb",
        "ram": "ram_gb",
        "disk": "storage_gb",
        "drive": "storage_gb",
        "display": "screen_inches",
        "screen": "screen_inches",
        "monitor": "screen_inches",
        "weight": "weight_kg",
        "battery": "battery_hours",
        "rating": "avg_rating",
        "reviews": "num_reviews",
        "gpu": "gpu_type",
        "graphics": "gpu_type",
        "storage": "storage_gb",
        "os": "os_version",
        "keyboard": "backlit_keyboard",
    }
    if user_lower in aliases:
        target = aliases[user_lower]
        if target in valid_fields:
            return target

    for f in valid_fields:
        if user_lower in f.lower() or f.lower() in user_lower:
            return f

    best_ratio, best_field = 0.0, None
    for f in valid_fields:
        ratio = SequenceMatcher(None, user_lower, f.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_field = f
    return best_field if best_ratio >= 0.6 else None


# ---------------------------------------------------------------------------
# 1. filter_laptops
# ---------------------------------------------------------------------------

def _build_filter_laptops_tool() -> DomainToolSpec:
    schema = {
        "type": "function",
        "function": {
            "name": "filter_laptops",
            "description": (
                "Filter the laptop catalog by structured constraints. "
                "Returns laptops matching ALL specified criteria. "
                "Only specify constraints the user has mentioned. "
                "Values are fuzzy-matched so natural phrasing works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price_max": {
                        "type": "number",
                        "description": "Maximum price in USD",
                    },
                    "price_min": {
                        "type": "number",
                        "description": "Minimum price in USD",
                    },
                    "ram_min_gb": {
                        "type": "integer",
                        "description": "Minimum RAM in GB (e.g. 8, 16, 32)",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type",
                        "enum": ["integrated", "dedicated"],
                    },
                    "brand": {
                        "type": "string",
                        "description": "Brand name (e.g. HP, Dell, Lenovo, ASUS, Acer, Apple, MSI, Samsung)",
                    },
                    "weight_max_kg": {
                        "type": "number",
                        "description": "Maximum weight in kilograms",
                    },
                    "storage_min_gb": {
                        "type": "integer",
                        "description": "Minimum storage in GB",
                    },
                    "screen_min_inches": {
                        "type": "number",
                        "description": "Minimum screen size in inches",
                    },
                    "screen_max_inches": {
                        "type": "number",
                        "description": "Maximum screen size in inches",
                    },
                    "battery_min_hours": {
                        "type": "number",
                        "description": "Minimum battery life in hours",
                    },
                    "storage_type": {
                        "type": "string",
                        "description": "Storage type",
                        "enum": ["SSD", "HDD", "eMMC", "hybrid"],
                    },
                    "touchscreen": {
                        "type": "boolean",
                        "description": "Whether a touchscreen is required",
                    },
                    "backlit_keyboard": {
                        "type": "boolean",
                        "description": "Whether a backlit keyboard is required",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Field to sort results by",
                        "enum": _SORT_FIELDS,
                    },
                    "sort_order": {
                        "type": "string",
                        "description": "Sort direction",
                        "enum": ["asc", "desc"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10, max 50)",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }

    def _filter_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        sort_by = kwargs.pop("sort_by", None)
        sort_order = kwargs.pop("sort_order", "asc")
        limit = min(int(kwargs.pop("limit", 10) or 10), 50)

        constraints = _extract_constraints(kwargs)

        if not constraints:
            sample = ctx.catalog.head(limit)
            lines = [f"Showing first {len(sample)} laptops (no filters applied)."]
            for idx, row in sample.iterrows():
                pid = str(row.get("product_id", idx))
                lines.append(f"- {_laptop_one_liner(row, pid)}")
            return "\n".join(lines)

        resolved, corrections = _fuzzy_resolve_constraints(constraints, ctx.config)

        try:
            filtered = GenericFilter.apply(ctx.catalog, resolved, ctx.config.constraints)
        except Exception as exc:
            logger.warning("GenericFilter.apply failed: %s. Falling back to manual filter.", exc)
            filtered = _manual_filter(ctx.catalog, constraints)

        if sort_by and sort_by in filtered.columns and len(filtered) > 0:
            ascending = sort_order.lower() != "desc"
            filtered = filtered.sort_values(
                by=sort_by, ascending=ascending, na_position="last",
            )

        total = len(filtered)
        display = filtered.head(limit)

        lines: list[str] = []
        if corrections:
            lines.append("Note: " + "; ".join(corrections))
        if total == 0:
            lines.append("No laptops matched all specified constraints. Try relaxing some filters.")
            return "\n".join(lines)

        lines.append(f"Found {total} matching laptops{f' (showing top {limit})' if total > limit else ''}.")
        for idx, row in display.iterrows():
            pid = str(row.get("product_id", idx))
            lines.append(f"- {_laptop_one_liner(row, pid)}")
        return "\n".join(lines)

    return DomainToolSpec(
        name="filter_laptops",
        description=schema["function"]["description"],
        fn=_filter_fn,
        schema=schema,
    )


def _manual_filter(catalog: pd.DataFrame, constraints: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=catalog.index)
    op_map = {
        "price_max_usd": ("price", "lte"),
        "price_min": ("price", "gte"),
        "ram_min_gb": ("ram_gb", "gte"),
        "gpu_type": ("gpu_type", "eq"),
        "brand": ("brand", "eq"),
        "weight_max_kg": ("weight_kg", "lte"),
        "storage_min_gb": ("storage_gb", "gte"),
        "screen_min_inches": ("screen_inches", "gte"),
        "screen_max_inches": ("screen_inches", "lte"),
        "battery_min_hours": ("battery_hours", "gte"),
        "storage_type": ("storage_type", "eq"),
        "touchscreen": ("touchscreen", "eq"),
        "backlit_keyboard": ("backlit_keyboard", "eq"),
    }
    for cname, val in constraints.items():
        if cname not in op_map:
            continue
        col_name, op = op_map[cname]
        if col_name not in catalog.columns:
            continue
        col = catalog[col_name]
        if op == "eq":
            if isinstance(val, bool):
                mask &= col.fillna(False).astype(bool) == val
            else:
                mask &= col.astype(str).str.lower().fillna("") == str(val).lower()
        elif op == "gte":
            mask &= pd.to_numeric(col, errors="coerce").fillna(-np.inf) >= float(val)
        elif op == "lte":
            mask &= pd.to_numeric(col, errors="coerce").fillna(np.inf) <= float(val)
    return catalog[mask]


# ---------------------------------------------------------------------------
# 2. search_laptops
# ---------------------------------------------------------------------------

_DOMAIN_DIR = Path(__file__).resolve().parent

_EMBEDDINGS_STEM = "all_embedding_text"
_UNIFIED_EMB_PATH = _DOMAIN_DIR / "extras" / f"{_EMBEDDINGS_STEM}_embeddings.npy"
_IDS_PATH = _DOMAIN_DIR / "extras" / "embedding_product_ids.json"

_unified_embeddings: np.ndarray | None = None
_embedding_product_ids: list[str] | None = None


def _load_unified_embeddings() -> np.ndarray | None:
    global _unified_embeddings
    if _unified_embeddings is not None:
        return _unified_embeddings
    if not _UNIFIED_EMB_PATH.is_file():
        return None
    _unified_embeddings = np.load(_UNIFIED_EMB_PATH)
    return _unified_embeddings


def _load_embedding_product_ids() -> list[str] | None:
    global _embedding_product_ids
    if _embedding_product_ids is not None:
        return _embedding_product_ids
    if not _IDS_PATH.is_file():
        return None
    with open(_IDS_PATH) as f:
        _embedding_product_ids = _json.load(f)
    return _embedding_product_ids


def _embed_query(query: str, model: str = "text-embedding-3-large") -> np.ndarray:
    from shared.llm import call_embedding
    raw = call_embedding(query, model=model)
    emb = np.array(raw[0], dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _semantic_search(
    query: str,
    catalog: pd.DataFrame,
    id_column: str,
    top_k: int,
) -> list[tuple[Any, float]]:
    product_ids = _load_embedding_product_ids()
    emb_matrix = _load_unified_embeddings()
    if product_ids is None or emb_matrix is None:
        return []

    pid_to_emb_idx = {pid: i for i, pid in enumerate(product_ids)}
    query_emb = _embed_query(query)

    results: list[tuple[Any, float]] = []
    for idx in catalog.index:
        pid = str(catalog.at[idx, id_column]) if id_column in catalog.columns else str(idx)
        emb_idx = pid_to_emb_idx.get(pid)
        if emb_idx is None or emb_idx >= len(emb_matrix):
            continue
        vec = emb_matrix[emb_idx]
        if float(np.linalg.norm(vec)) == 0.0:
            continue
        sim = float(np.dot(vec, query_emb))
        if sim > 0:
            results.append((idx, sim))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def _build_search_laptops_tool() -> DomainToolSpec:
    description = (
        "Search the laptop catalog using natural language. "
        "Supports two modes: 'keyword' (TF-IDF text matching across name, brand, "
        "processor, GPU, storage type, display, OS, use cases, and description) "
        "and 'semantic' (embedding similarity using text-embedding-3-large). "
        "Supports optional structured constraints to narrow results before searching."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "search_laptops",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'lightweight gaming laptop under $1000')",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["keyword", "semantic"],
                        "description": (
                            "Search mode. 'keyword' uses TF-IDF text matching (good for exact "
                            "terms like brand names, processor models). 'semantic' uses embedding "
                            "similarity (good for conceptual queries). Default: keyword."
                        ),
                    },
                    "brand": {"type": "string", "description": "Optional: filter by brand."},
                    "price_max": {"type": "number", "description": "Optional: max price in USD."},
                    "gpu_type": {
                        "type": "string",
                        "description": "Optional: filter by GPU type.",
                        "enum": ["integrated", "dedicated"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10, max 30).",
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    TF_IDF_COLS = [c for c in _SEARCH_FIELDS]

    def _row_blob(row: pd.Series, cols: list[str]) -> str:
        parts = []
        for col in cols:
            v = row.get(col) if col in row.index else None
            if _is_null(v):
                continue
            items = _safe_list(v)
            if items:
                parts.append(" ".join(str(x) for x in items))
            else:
                parts.append(str(v))
        return " ".join(parts)

    def _tfidf_search(
        query: str,
        filtered: pd.DataFrame,
        searchable: list[str],
        limit: int,
    ) -> list[dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        idf_sample = (
            filtered.sample(n=min(5000, len(filtered)), random_state=42)
            if len(filtered) > 5000 else filtered
        )
        N = len(idf_sample)
        doc_freq: Counter = Counter()
        for _, row in idf_sample.iterrows():
            blob = _normalize_text(_row_blob(row, searchable))
            for tok in query_tokens:
                if tok in blob:
                    doc_freq[tok] += 1

        idf = {tok: float(np.log((N + 1) / (doc_freq[tok] + 1)) + 1.0) for tok in query_tokens}

        pop_col = "num_reviews"
        has_popularity = pop_col in filtered.columns
        if has_popularity:
            pop_vals = pd.to_numeric(filtered[pop_col], errors="coerce")
            pop_max = pop_vals.max()
            pop_min = pop_vals.min()
            pop_range = max(pop_max - pop_min, 1)

        results = []
        for idx, row in filtered.iterrows():
            blob = _normalize_text(_row_blob(row, searchable))
            if not blob:
                continue
            blob_tokens = set(_tokenize(blob))
            score = 0.0
            matched = []
            for tok in query_tokens:
                w = idf.get(tok, 1.0)
                if tok in blob_tokens:
                    score += w
                    matched.append(tok)
                elif tok in blob:
                    score += 0.5 * w
                    matched.append(f"{tok}~")
            if score > 0:
                if has_popularity:
                    pop_val = _safe_val(row, pop_col)
                    if pop_val is not None:
                        try:
                            pop_norm = (float(pop_val) - pop_min) / pop_range
                            score *= (1.0 + 0.2 * pop_norm)
                        except (ValueError, TypeError):
                            pass
                results.append({"idx": idx, "score": score, "matched": matched})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        if not query or not query.strip():
            return "Please provide a search query."

        mode = kwargs.get("mode", "keyword")
        limit = min(max(1, int(kwargs.get("limit", 10) or 10)), 30)

        constraints = _extract_constraints(kwargs)
        filtered = ctx.catalog
        corrections = []
        if constraints:
            resolved, corrections = _fuzzy_resolve_constraints(constraints, ctx.config)
            try:
                filtered = GenericFilter.apply(filtered, resolved, ctx.config.constraints)
            except Exception:
                pass

        if len(filtered) == 0:
            return "No laptops matched the constraints."

        if mode == "semantic":
            if not _UNIFIED_EMB_PATH.is_file():
                return (
                    "Semantic search unavailable (no embeddings file). "
                    "Use mode='keyword' instead."
                )

            id_col = ctx.config.id_column
            sem_results = _semantic_search(query, filtered, id_col, top_k=limit)
            if not sem_results:
                return f"No laptops matched the semantic query '{query}'."

            lines = [
                f"Semantic search results for '{query}' ({len(sem_results)} results):"
            ]
            if corrections:
                lines.append("Corrections: " + "; ".join(corrections))
            for rank, (idx, score) in enumerate(sem_results, 1):
                row = filtered.loc[idx]
                pid = str(row.get("product_id", idx))
                lines.append(f"{rank}. {_laptop_one_liner(row, pid)} [sim={score:.3f}]")
            return "\n".join(lines)

        searchable = [c for c in TF_IDF_COLS if c in filtered.columns]
        results = _tfidf_search(query, filtered, searchable, limit)

        if not results:
            return f"No laptops matched the query '{query}'."

        lines = [f"Search results for '{query}' ({len(results)} matches):"]
        if corrections:
            lines.append("Corrections: " + "; ".join(corrections))
        for rank, item in enumerate(results, 1):
            row = filtered.loc[item["idx"]]
            pid = str(row.get("product_id", item["idx"]))
            lines.append(f"  {rank}. {_laptop_one_liner(row, pid)}")
        return "\n".join(lines)

    return DomainToolSpec(name="search_laptops", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# 3. get_laptop_details
# ---------------------------------------------------------------------------

def _build_get_laptop_details_tool() -> DomainToolSpec:
    description = (
        "Get detailed information about a specific laptop by its product ID. "
        "Returns all available specs including price, RAM, storage, screen, "
        "GPU, battery, weight, processor, display, ports, and ratings."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "get_laptop_details",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID (e.g. 'B0BM41YRL6')",
                    },
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
        },
    }

    _DISPLAY_ORDER = [
        ("product_id", "Product ID", "str"),
        ("name", "Name", "str"),
        ("brand", "Brand", "str"),
        ("price", "Price", "price"),
        ("ram_gb", "RAM", "gb"),
        ("storage_gb", "Storage", "gb"),
        ("storage_type", "Storage Type", "str"),
        ("processor", "Processor", "str"),
        ("screen_inches", "Screen Size", "inches"),
        ("display_type", "Display Type", "str"),
        ("touchscreen", "Touchscreen", "bool"),
        ("refresh_rate_hz", "Refresh Rate", "hz"),
        ("gpu_type", "GPU Type", "str"),
        ("weight_kg", "Weight", "kg"),
        ("battery_hours", "Battery Life", "hours"),
        ("backlit_keyboard", "Backlit Keyboard", "bool"),
        ("numeric_keypad", "Numeric Keypad", "bool"),
        ("webcam_resolution", "Webcam Resolution", "str"),
        ("os_version", "OS", "str"),
        ("avg_rating", "Rating", "rating"),
        ("num_reviews", "Number of Reviews", "int"),
        ("ports", "Ports", "list"),
        ("use_cases", "Use Cases", "list"),
        ("description", "Description", "desc"),
    ]

    def _detail_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        pid = kwargs.get("product_id")
        if not pid:
            return "Missing required product_id."
        pid = str(pid).strip()

        row = ctx.get_product(pid)
        if row is None:
            return f"Product '{pid}' not found in catalog."

        name = _safe_val(row, "name") or "Unknown"
        brand = _safe_val(row, "brand") or "Unknown"
        lines = [f"=== {brand} — {name} ===", ""]

        for col, label, fmt in _DISPLAY_ORDER:
            if col in ("name",):
                continue
            val = _safe_val(row, col)
            if val is None and fmt != "str":
                continue

            if fmt == "price":
                lines.append(f"  {label}: {_fmt_price(val)}")
            elif fmt == "gb":
                lines.append(f"  {label}: {_fmt_int(val)} GB")
            elif fmt == "inches":
                lines.append(f'  {label}: {_fmt_float(val)}"')
            elif fmt == "kg":
                lines.append(f"  {label}: {_fmt_float(val, 2)} kg")
            elif fmt == "hours":
                lines.append(f"  {label}: {_fmt_float(val)} hours")
            elif fmt == "hz":
                lines.append(f"  {label}: {_fmt_int(val)} Hz")
            elif fmt == "bool":
                lines.append(f"  {label}: {_fmt_bool(val)}")
            elif fmt == "rating":
                nr = _fmt_int(_safe_val(row, "num_reviews"))
                lines.append(f"  {label}: {_fmt_float(val)}/5 ({nr} reviews)")
            elif fmt == "int":
                continue
            elif fmt == "list":
                items = _safe_list(val)
                if items:
                    lines.append(f"  {label}: {', '.join(str(x) for x in items)}")
            elif fmt == "desc":
                s = str(val)
                if s:
                    if len(s) > 300:
                        s = s[:300] + "..."
                    lines.append(f"  {label}: {s}")
            else:
                if val is not None:
                    lines.append(f"  {label}: {val}")

        return "\n".join(lines)

    return DomainToolSpec(
        name="get_laptop_details",
        description=description,
        fn=_detail_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# 4. get_catalog_stats
# ---------------------------------------------------------------------------

def _build_get_catalog_stats_tool() -> DomainToolSpec:
    description = (
        "Get summary statistics about the laptop catalog: "
        "total products, price range, brands, RAM options, GPU types, "
        "weight range, and screen sizes."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "get_catalog_stats",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }

    def _stats_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        cat = ctx.catalog
        brands = sorted(cat["brand"].dropna().unique().tolist()) if "brand" in cat.columns else []
        ram_opts = sorted(pd.to_numeric(cat["ram_gb"], errors="coerce").dropna().unique().tolist()) if "ram_gb" in cat.columns else []
        gpu_counts = cat["gpu_type"].dropna().value_counts().to_dict() if "gpu_type" in cat.columns else {}

        lines = [
            "Catalog Summary:",
            f"  Total products: {len(cat)}",
        ]
        if "price" in cat.columns:
            p = pd.to_numeric(cat["price"], errors="coerce").dropna()
            if len(p) > 0:
                lines.append(
                    f"  Price range: ${p.min():.0f} – ${p.max():.0f} "
                    f"(median ${p.median():.0f})"
                )
        if brands:
            lines.append(f"  Brands ({len(brands)}): {', '.join(str(b) for b in brands[:15])}")
        if ram_opts:
            lines.append(f"  RAM options (GB): {[int(r) for r in ram_opts]}")
        if gpu_counts:
            lines.append(f"  GPU types: {gpu_counts}")
        if "weight_kg" in cat.columns:
            w = pd.to_numeric(cat["weight_kg"], errors="coerce").dropna()
            if len(w) > 0:
                lines.append(f"  Weight range: {w.min():.1f} – {w.max():.1f} kg")
        if "screen_inches" in cat.columns:
            s = pd.to_numeric(cat["screen_inches"], errors="coerce").dropna()
            if len(s) > 0:
                lines.append(f'  Screen sizes: {s.min():.1f} – {s.max():.1f}"')
        if "battery_hours" in cat.columns:
            b = pd.to_numeric(cat["battery_hours"], errors="coerce").dropna()
            if len(b) > 0:
                lines.append(f"  Battery range: {b.min():.1f} – {b.max():.1f} hours")
        if "storage_type" in cat.columns:
            st_counts = cat["storage_type"].dropna().value_counts().to_dict()
            lines.append(f"  Storage types: {st_counts}")

        return "\n".join(lines)

    return DomainToolSpec(
        name="get_catalog_stats",
        description=description,
        fn=_stats_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_domain_specific_tools() -> list[DomainToolSpec]:
    return [
        _build_filter_laptops_tool(),
        _build_search_laptops_tool(),
        _build_get_laptop_details_tool(),
        _build_get_catalog_stats_tool(),
    ]
