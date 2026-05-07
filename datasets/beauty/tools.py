"""Domain-specific tools for Sephora beauty products."""

from __future__ import annotations

import ast
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.config import ConstraintOp, AttrType
from simulation.tools import (
    DomainCatalogContext,
    DomainToolSpec,
    _fuzzy_resolve_constraints,
)
from shared.filter import GenericFilter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _normalize_bool(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return v


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _normalize_text(text)) if len(t) >= 2]


def _fmt_price(v: Any) -> str:
    if _is_null(v):
        return "N/A"
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return str(v)


def _fmt_int(v: Any) -> str:
    if _is_null(v):
        return "N/A"
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return str(v)


def _fmt_float(v: Any, digits: int = 1) -> str:
    if _is_null(v):
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _fmt_bool(v: Any) -> str:
    if _is_null(v):
        return "N/A"
    b = _normalize_bool(v)
    if isinstance(b, bool):
        return "Yes" if b else "No"
    return str(v)


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


def _category_path(row: pd.Series) -> str:
    parts = []
    for col in ["primary_category", "secondary_category", "tertiary_category"]:
        v = row.get(col) if col in row.index else None
        if not _is_null(v) and str(v).strip():
            parts.append(str(v).strip())
    return " > ".join(parts) if parts else "Uncategorized"


def _product_line(row: pd.Series) -> str:
    pid = str(row.name) if hasattr(row, "name") else "?"
    name = "" if _is_null(row.get("product_name")) else str(row["product_name"])
    brand = "" if _is_null(row.get("brand_name")) else str(row["brand_name"])
    cat = _category_path(row)
    price = _fmt_price(row.get("price_usd"))
    rating = _fmt_float(row.get("rating"))
    reviews = _fmt_int(row.get("reviews"))
    loves = _fmt_int(row.get("loves_count"))
    oos = _normalize_bool(row.get("out_of_stock"))
    stock = "out of stock" if oos is True else "in stock"
    return (
        f"- {name} ({pid}) — {brand}; {cat}; {price}; "
        f"rating {rating}; {reviews} reviews; {loves} loves; {stock}"
    )


def _extract_constraints(kwargs: dict) -> dict:
    mapping = {
        "brand_name": "brand_name",
        "primary_category": "primary_category",
        "secondary_category": "secondary_category",
        "tertiary_category": "tertiary_category",
        "variation_value": "variation_value",
        "price_usd_max": "price_usd_max",
        "price_usd_min": "price_usd_min",
        "limited_edition": "limited_edition",
        "new": "new",
        "online_only": "online_only",
        "out_of_stock": "out_of_stock",
        "sephora_exclusive": "sephora_exclusive",
        "highlights": "highlights_includes",
    }
    constraints = {}
    for param, cname in mapping.items():
        v = kwargs.get(param)
        if v is None:
            continue
        if param in ("limited_edition", "new", "online_only", "out_of_stock", "sephora_exclusive"):
            v = _normalize_bool(v)
        elif param in ("price_usd_max", "price_usd_min"):
            try:
                v = float(v)
            except Exception:
                continue
        elif param in ("brand_name", "secondary_category", "tertiary_category", "variation_value"):
            if isinstance(v, str):
                v = [v]
        elif param == "highlights":
            if isinstance(v, str):
                v = [v]
        constraints[cname] = v
    return constraints


# ---------------------------------------------------------------------------
# 1. filter_beauty_products
# ---------------------------------------------------------------------------

def _build_filter_beauty_products_tool() -> DomainToolSpec:
    description = (
        "Filter the beauty product catalog using structured constraints. "
        "Supports filtering by brand, category hierarchy, price range, "
        "variation, highlights/tags, and boolean flags. "
        "Returns matching products sorted by the chosen field."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "filter_beauty_products",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_name": {
                        "type": "string",
                        "description": "Filter by brand name (fuzzy matched). Pass a single brand.",
                    },
                    "primary_category": {
                        "type": "string",
                        "description": (
                            "Exact primary category: Makeup, Skincare, Fragrance, Hair, "
                            "Tools & Brushes, Bath & Body, Mini Size, Gifts."
                        ),
                    },
                    "secondary_category": {
                        "type": "string",
                        "description": "Filter by secondary category (e.g., Moisturizers, Lip, Eye, Face).",
                    },
                    "tertiary_category": {
                        "type": "string",
                        "description": "Filter by tertiary category (e.g., Lipstick, Mascara, Face Primer, Face Serums).",
                    },
                    "variation_value": {
                        "type": "string",
                        "description": "Filter by specific variant value such as shade name, size, or scent (fuzzy matched).",
                    },
                    "price_usd_max": {
                        "type": "number",
                        "description": "Maximum price in USD.",
                    },
                    "price_usd_min": {
                        "type": "number",
                        "description": "Minimum price in USD.",
                    },
                    "highlights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Filter by product highlights/tags. Products matching ANY of the "
                            "given tags are returned. Common values include: Vegan, Cruelty-Free, "
                            "Clean at Sephora, Fragrance Free, Gluten Free, Sulfate Free, "
                            "Paraben Free, Silicone Free, Without Parabens, Natural, Oil Free. "
                            "Use list_catalog_values(field_name='highlights') to discover all tags."
                        ),
                    },
                    "limited_edition": {
                        "type": "boolean",
                        "description": "Filter for limited edition products.",
                    },
                    "new": {
                        "type": "boolean",
                        "description": "Filter for new products.",
                    },
                    "online_only": {
                        "type": "boolean",
                        "description": "Filter for online-only products.",
                    },
                    "out_of_stock": {
                        "type": "boolean",
                        "description": "Filter by stock status (true = out of stock only).",
                    },
                    "sephora_exclusive": {
                        "type": "boolean",
                        "description": "Filter for Sephora exclusive products.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["loves_count", "rating", "reviews", "price_usd", "product_name"],
                        "description": "Field to sort results by (default: loves_count).",
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (default: desc for popularity metrics, asc for price/name).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10).",
                        "minimum": 1,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }

    def _fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        constraints = _extract_constraints(kwargs)
        if not constraints:
            return "Please provide at least one filter constraint."

        resolved, corrections = _fuzzy_resolve_constraints(constraints, ctx.config)

        try:
            filtered = GenericFilter.apply(ctx.catalog, resolved, ctx.config.constraints)
        except Exception:
            filtered = ctx.catalog.copy()

        total = len(filtered)
        if total == 0:
            msg = "No matching beauty products found."
            if corrections:
                msg += " Corrections: " + "; ".join(corrections)
            return msg

        sort_by = kwargs.get("sort_by", "loves_count")
        sort_order = kwargs.get("sort_order")
        limit = min(max(1, int(kwargs.get("limit", 10) or 10)), 100)

        if sort_by not in {"loves_count", "rating", "reviews", "price_usd", "product_name"}:
            sort_by = "loves_count"
        if sort_order not in {"asc", "desc"}:
            sort_order = "desc" if sort_by in {"loves_count", "rating", "reviews"} else "asc"

        if sort_by in filtered.columns:
            if sort_by in {"loves_count", "rating", "reviews", "price_usd"}:
                filtered = filtered.assign(
                    _sort=pd.to_numeric(filtered[sort_by], errors="coerce")
                ).sort_values("_sort", ascending=(sort_order == "asc"), na_position="last")
            else:
                filtered = filtered.sort_values(sort_by, ascending=(sort_order == "asc"), na_position="last")

        top = filtered.head(limit)
        lines = [f"Found {total} matching beauty products."]
        if corrections:
            lines.append("Corrections: " + "; ".join(corrections))
        for _, row in top.iterrows():
            lines.append(_product_line(row))
        if total > limit:
            lines.append(f"Showing top {limit} of {total} results sorted by {sort_by} ({sort_order}).")
        return "\n".join(lines)

    return DomainToolSpec(name="filter_beauty_products", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# 2. search_beauty_products
# ---------------------------------------------------------------------------

_DOMAIN_DIR = Path(__file__).resolve().parent

# Unified semantic index: Stage 3f builds one matrix from all embedding_text fields.
_EMBEDDINGS_STEM = "all_embedding_text"
_UNIFIED_EMB_PATH = _DOMAIN_DIR / f"{_EMBEDDINGS_STEM}_embeddings.npy"
_CONFIG_PATH = _DOMAIN_DIR / "config.json"

_unified_embeddings: np.ndarray | None = None
_embedding_product_ids: list[str] | None = None


def _embedding_field_pairs_from_config(config_path: Path) -> list[tuple[str, str]]:
    """(column_name, display_name) for attributes with embedding_field=True."""
    try:
        from shared.config import DomainConfig

        cfg = DomainConfig.load(str(config_path))
    except Exception:
        return []
    return [(a.name, a.display_name) for a in cfg.attributes if a.embedding_field]


def _unified_embedding_index_description() -> str:
    pairs = _embedding_field_pairs_from_config(_CONFIG_PATH)
    if not pairs:
        return (
            "Semantic search uses one embedding per product, built from catalog fields "
            "marked embedding_text in the domain config (`embedding_field`)."
        )
    formatted = "; ".join(f"`{name}` ({dname})" for name, dname in pairs)
    return (
        "Semantic mode searches a unified embedding built by concatenating these fields "
        f"(newline-separated \"field: text\" lines): {formatted}."
    )


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
    ids_path = _DOMAIN_DIR / "embedding_product_ids.json"
    if not ids_path.is_file():
        return None
    import json
    with open(ids_path) as f:
        _embedding_product_ids = json.load(f)
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
    """Return (index, score) pairs from unified embedding cosine search."""
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


def _build_search_beauty_products_tool() -> DomainToolSpec:
    _emb_index_blurb = _unified_embedding_index_description()
    description = (
        "Search the beauty product catalog using natural language. "
        "Supports two modes: 'keyword' (TF-IDF text matching) and 'semantic' "
        "(unified embedding similarity over long catalog text). "
        "Supports optional structured constraints to narrow results before searching. "
        "Search only considers rows that have usable source data (non-empty text in the "
        "indexed fields for keyword mode, or an entry in the embedding index for semantic mode); "
        "not every catalog row is searchable. "
        f"{_emb_index_blurb}"
    )

    TF_IDF_COLS = [
        "product_name", "brand_name", "primary_category", "secondary_category",
        "tertiary_category", "highlights", "ingredients", "variation_type",
        "variation_value", "variation_desc", "size",
    ]

    schema_properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "Natural language search query (e.g., 'hydrating moisturizer for dry skin').",
        },
        "mode": {
            "type": "string",
            "enum": ["keyword", "semantic"],
            "description": (
                "Search mode. 'keyword' uses TF-IDF text matching (good for exact terms "
                "like brand names, specific ingredients). 'semantic' uses embedding similarity "
                "against the unified all_embedding_text vector (covers the fields noted in "
                "the tool description). Default: keyword."
            ),
        },
        "brand_name": {"type": "string", "description": "Optional: filter by brand."},
        "primary_category": {"type": "string", "description": "Optional: filter by primary category."},
        "price_usd_max": {"type": "number", "description": "Optional: max price."},
        "limit": {
            "type": "integer",
            "description": "Max results to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
    }

    schema = {
        "type": "function",
        "function": {
            "name": "search_beauty_products",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": schema_properties,
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

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
                results.append({"idx": idx, "score": score, "matched": matched})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        if not query or not query.strip():
            return "Please provide a search query."

        mode = kwargs.get("mode", "keyword")
        limit = min(max(1, int(kwargs.get("limit", 10) or 10)), 50)

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
            return "No products matched the constraints."

        if mode == "semantic":
            if not _UNIFIED_EMB_PATH.is_file():
                return (
                    "Semantic search unavailable (no unified embeddings at "
                    f"{_UNIFIED_EMB_PATH.name}). Run notebook Stage 3f or use "
                    "mode='keyword' instead."
                )

            id_col = ctx.config.id_column
            sem_results = _semantic_search(query, filtered, id_col, top_k=limit)

            if not sem_results:
                return f"No products matched the semantic query '{query}'."

            lines = [
                f"Semantic search results for '{query}' "
                f"(unified embedding {_EMBEDDINGS_STEM}, {len(sem_results)} results):"
            ]
            if corrections:
                lines.append("Corrections: " + "; ".join(corrections))

            for rank, (idx, score) in enumerate(sem_results, 1):
                row = filtered.loc[idx]
                pid = str(idx)
                name = "" if _is_null(row.get("product_name")) else str(row["product_name"])
                brand = "" if _is_null(row.get("brand_name")) else str(row["brand_name"])
                price = _fmt_price(row.get("price_usd"))
                cat = _category_path(row)
                rating_s = _fmt_float(row.get("rating"))
                reviews_s = _fmt_int(row.get("reviews"))
                loves_s = _fmt_int(row.get("loves_count"))
                lines.append(
                    f"{rank}. {name} ({pid}) — {brand}; {cat}; {price}; "
                    f"rating {rating_s}; {reviews_s} reviews; {loves_s} loves "
                    f"[sim={score:.3f}]"
                )

            return "\n".join(lines)

        # Default: keyword / TF-IDF mode
        searchable = [c for c in TF_IDF_COLS if c in filtered.columns]
        query_tokens = _tokenize(query)
        if not query_tokens:
            return "Please provide a more specific search query."

        results = _tfidf_search(query, filtered, searchable, limit)

        if not results:
            return f"No products matched the query '{query}'."

        lines = [f"Search results for '{query}' ({len(results)} matches, showing top {min(limit, len(results))}):"]
        if corrections:
            lines.append("Corrections: " + "; ".join(corrections))

        for rank, item in enumerate(results, 1):
            row = filtered.loc[item["idx"]]
            pid = str(item["idx"])
            name = "" if _is_null(row.get("product_name")) else str(row["product_name"])
            brand = "" if _is_null(row.get("brand_name")) else str(row["brand_name"])
            price = _fmt_price(row.get("price_usd"))
            cat = _category_path(row)
            rating_s = _fmt_float(row.get("rating"))
            reviews_s = _fmt_int(row.get("reviews"))
            loves_s = _fmt_int(row.get("loves_count"))
            oos = _normalize_bool(row.get("out_of_stock"))
            stock = "out of stock" if oos is True else "in stock"
            lines.append(
                f"{rank}. {name} ({pid}) — {brand}; {cat}; {price}; "
                f"rating {rating_s}; {reviews_s} reviews; {loves_s} loves; {stock}"
            )

        return "\n".join(lines)

    return DomainToolSpec(name="search_beauty_products", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# 3. get_product_details
# ---------------------------------------------------------------------------

def _build_get_product_details_tool() -> DomainToolSpec:
    description = (
        "Retrieve the full catalog record for a beauty product by its exact product name. "
        "Use the product name shown in search/filter results."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The exact product name as shown in search/filter results (e.g., 'Black Opium Eau de Parfum').",
                    },
                },
                "required": ["product_name"],
                "additionalProperties": False,
            },
        },
    }

    FIELD_ORDER = [
        "product_name", "brand_name",
        "primary_category", "secondary_category", "tertiary_category",
        "price_usd", "sale_price_usd", "value_price_usd",
        "rating", "reviews", "loves_count",
        "size", "variation_type", "variation_value", "variation_desc",
        "highlights", "ingredients",
        "limited_edition", "new", "online_only", "out_of_stock", "sephora_exclusive",
        "child_count", "child_max_price", "child_min_price",
    ]

    LABELS = {
        "product_name": "Product",
        "brand_name": "Brand",
        "primary_category": "Primary Category",
        "secondary_category": "Secondary Category",
        "tertiary_category": "Tertiary Category",
        "price_usd": "Price",
        "sale_price_usd": "Sale Price",
        "value_price_usd": "Value Price",
        "rating": "Rating",
        "reviews": "Reviews",
        "loves_count": "Loves",
        "size": "Size",
        "variation_type": "Variation Type",
        "variation_value": "Variation Value",
        "variation_desc": "Variation Description",
        "highlights": "Highlights",
        "ingredients": "Ingredients",
        "limited_edition": "Limited Edition",
        "new": "New",
        "online_only": "Online Only",
        "out_of_stock": "Out of Stock",
        "sephora_exclusive": "Sephora Exclusive",
        "child_count": "Child Products",
        "child_max_price": "Child Max Price",
        "child_min_price": "Child Min Price",
    }

    BOOL_FIELDS = {"limited_edition", "new", "online_only", "out_of_stock", "sephora_exclusive"}
    PRICE_FIELDS = {"price_usd", "sale_price_usd", "value_price_usd", "child_max_price", "child_min_price"}

    def _fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        pid = kwargs.get("product_name")
        if not pid:
            return "Missing required product_name."
        pid = str(pid).strip()

        row = ctx.get_product(pid)
        if row is None:
            return f"Product not found: {pid}"

        lines = []
        brand = row.get("brand_name") if "brand_name" in row.index else None
        lines.append(f"Product: {pid}")
        if not _is_null(brand):
            lines.append(f"Brand: {brand}")
        lines.append(f"Category: {_category_path(row)}")

        for col in FIELD_ORDER:
            if col in ("product_name", "brand_name", "primary_category", "secondary_category", "tertiary_category"):
                continue
            if col not in row.index:
                continue
            v = row[col]
            if _is_null(v):
                continue

            label = LABELS.get(col, col.replace("_", " ").title())

            if col in BOOL_FIELDS:
                lines.append(f"{label}: {_fmt_bool(v)}")
            elif col in PRICE_FIELDS:
                lines.append(f"{label}: {_fmt_price(v)}")
            elif col == "rating":
                lines.append(f"{label}: {_fmt_float(v)}/5.0")
            elif col in ("reviews", "loves_count", "child_count"):
                lines.append(f"{label}: {_fmt_int(v)}")
            elif col == "highlights":
                items = _safe_list(v)
                lines.append(f"{label}: {', '.join(str(x) for x in items)}" if items else f"{label}: {v}")
            elif col == "ingredients":
                s = str(v)
                if len(s) > 300:
                    s = s[:300] + "..."
                lines.append(f"{label}: {s}")
            else:
                lines.append(f"{label}: {v}")

        return "\n".join(lines)

    return DomainToolSpec(name="get_product_details", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# 4. list_catalog_values
# ---------------------------------------------------------------------------

def _build_list_catalog_values_tool() -> DomainToolSpec:
    description = (
        "Discover distinct values for a catalog field (e.g., brands, categories, "
        "highlights) with counts. Supports fuzzy field name matching."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "list_catalog_values",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "description": (
                            "Catalog field to inspect (e.g., brand_name, primary_category, "
                            "secondary_category, tertiary_category, highlights, variation_type)."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional substring to filter values.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max values to return (default 20).",
                    },
                },
                "required": ["field_name"],
                "additionalProperties": False,
            },
        },
    }

    ALIASES = {
        "brand": "brand_name",
        "brands": "brand_name",
        "category": "primary_category",
        "categories": "primary_category",
        "subcategory": "secondary_category",
        "highlight": "highlights",
        "ingredient": "ingredients",
        "price": "price_usd",
        "variation": "variation_type",
    }

    def _resolve_field(catalog: pd.DataFrame, raw: str) -> str | None:
        norm = raw.strip().lower().replace(" ", "_")
        if norm in catalog.columns:
            return norm
        if norm in ALIASES and ALIASES[norm] in catalog.columns:
            return ALIASES[norm]
        for col in catalog.columns:
            if norm in col or col in norm:
                return col
        best_ratio, best_col = 0.0, None
        for col in catalog.columns:
            ratio = SequenceMatcher(None, norm, col).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col
        return best_col if best_ratio >= 0.6 else None

    def _get_set_valued_fields(ctx: DomainCatalogContext) -> set[str]:
        return {
            a.name for a in ctx.config.attributes
            if a.attr_type == AttrType.SET_VALUED
        }

    def _fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        raw_field = kwargs.get("field_name", "")
        if not raw_field:
            return "Please provide a field_name."

        field = _resolve_field(ctx.catalog, str(raw_field))
        if field is None:
            avail = ", ".join(sorted(ctx.catalog.columns)[:20])
            return f"Unknown field '{raw_field}'. Available: {avail}"

        query = kwargs.get("query")
        limit = min(max(1, int(kwargs.get("limit", 20) or 20)), 200)

        set_valued_fields = _get_set_valued_fields(ctx)

        counts: Counter = Counter()
        display_map: dict[str, str] = {}

        if field in set_valued_fields:
            for v in ctx.catalog[field]:
                for item in _safe_list(v):
                    key = _normalize_text(item)
                    if key:
                        counts[key] += 1
                        if key not in display_map:
                            display_map[key] = str(item).strip()
        else:
            for v in ctx.catalog[field]:
                if _is_null(v):
                    continue
                key = _normalize_text(v)
                if key:
                    counts[key] += 1
                    if key not in display_map:
                        display_map[key] = str(v).strip()

        if not counts:
            return f"No values found for '{field}'."

        matched = list(counts.keys())
        note = ""
        if query and query.strip():
            q = _normalize_text(query)
            exact = [k for k in matched if q in k or k in q]
            if exact:
                matched = exact
            else:
                scored = [(SequenceMatcher(None, q, k).ratio(), k) for k in matched]
                matched = [k for ratio, k in sorted(scored, reverse=True) if ratio >= 0.5]
                if matched:
                    note = f" (fuzzy match for '{query}')"

        if not matched:
            return f"No values for '{field}' matching '{query}'."

        matched.sort(key=lambda k: (-counts[k], display_map.get(k, k).lower()))
        top = matched[:limit]

        total_distinct = len(counts)
        lines = [f"Values for '{field}'{note} ({total_distinct} distinct):"]
        for k in top:
            lines.append(f"  {display_map.get(k, k)}: {counts[k]}")
        if len(matched) > limit:
            lines.append(f"  ... and {len(matched) - limit} more.")

        return "\n".join(lines)

    return DomainToolSpec(name="list_catalog_values", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_domain_specific_tools() -> list[DomainToolSpec]:
    return [
        _build_filter_beauty_products_tool(),
        _build_search_beauty_products_tool(),
        _build_get_product_details_tool(),
        _build_list_catalog_values_tool(),
    ]
