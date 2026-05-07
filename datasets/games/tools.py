"""Domain-specific tools for the video games recommendation benchmark.

Provides four tools tailored to the Steam game catalog:
  1. filter_games        — structured constraint filtering
  2. search_games        — keyword (TF-IDF) or semantic (embedding) search
  3. get_game_details    — full detail lookup by game title
  4. list_catalog_values — discover valid values for a catalog field

Entry point: ``build_domain_specific_tools()`` returns ``list[DomainToolSpec]``.
"""

from __future__ import annotations

import ast
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


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


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _normalize_text(text)) if len(t) >= 2]


def _fmt_price(v: Any) -> str:
    if _is_null(v):
        return "N/A"
    try:
        p = float(v)
        if p == 0.0:
            return "Free to Play"
        return f"${p:,.2f}"
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


def _fmt_date(v: Any) -> str:
    if _is_null(v):
        return "N/A"
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except Exception:
        return str(v)


def _fmt_playtime(minutes: Any) -> str:
    if _is_null(v := minutes):
        return "N/A"
    try:
        m = int(float(v))
    except (TypeError, ValueError):
        return "N/A"
    if m == 0:
        return "0h 0m"
    hours, mins = divmod(m, 60)
    if hours == 0:
        return f"{mins}m"
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


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


def _list_to_str(v: Any, max_items: int = 0) -> str:
    items = _safe_list(v)
    if not items:
        return "N/A"
    if max_items > 0 and len(items) > max_items:
        return ", ".join(str(x) for x in items[:max_items]) + f" (+{len(items) - max_items} more)"
    return ", ".join(str(x) for x in items)


def _review_summary(pos: Any, neg: Any) -> str:
    try:
        p, n = int(pos), int(neg)
    except (TypeError, ValueError):
        return "No reviews"
    total = p + n
    if total == 0:
        return "No reviews"
    pct = p / total * 100
    if total >= 1_000_000:
        count_str = f"{total / 1_000_000:.1f}M"
    elif total >= 1_000:
        count_str = f"{total / 1_000:.1f}K"
    else:
        count_str = str(total)
    return f"{pct:.0f}% positive ({count_str} reviews)"


def _product_line(row: pd.Series) -> str:
    name = str(row.name) if hasattr(row, "name") else ""
    price = _fmt_price(row.get("price"))
    genres = _list_to_str(row.get("genres"), max_items=3)
    platforms = _list_to_str(row.get("supported_platforms"), max_items=3)
    pos = 0 if _is_null(row.get("positive")) else int(row["positive"])
    neg = 0 if _is_null(row.get("negative")) else int(row.get("negative", 0))
    review = _review_summary(pos, neg)
    release = _fmt_date(row.get("release_date"))
    return (
        f"- {name} — {price}; {genres}; {review}; "
        f"platforms: {platforms}; released: {release}"
    )


def _extract_constraints(kwargs: dict) -> dict:
    mapping = {
        "genres": "genres_includes",
        "categories": "categories_includes",
        "price_max": "price_max",
        "supported_platforms": "supported_platforms_includes",
        "supported_languages": "supported_languages_includes",
        "full_audio_languages": "full_audio_languages_includes",
        "developer": "developer",
        "publishers": "publishers_includes",
        "positive_min": "positive_min",
        "recommendations_min": "recommendations_min",
        "release_date_min": "release_date_min",
    }
    constraints: dict[str, Any] = {}
    for param, cname in mapping.items():
        v = kwargs.get(param)
        if v is None:
            continue
        if param in ("price_max",):
            try:
                v = float(v)
            except Exception:
                continue
        elif param in ("positive_min", "recommendations_min"):
            try:
                v = int(v)
            except Exception:
                continue
        elif param == "release_date_min":
            try:
                v = pd.Timestamp(v)
            except Exception:
                v = str(v)
        elif param in ("developer",):
            if isinstance(v, str):
                v = [v]
        constraints[cname] = v
    return constraints


# ---------------------------------------------------------------------------
# Embedding infrastructure (mirrors Sephora pattern)
# ---------------------------------------------------------------------------

_DOMAIN_DIR = Path(__file__).resolve().parent
_EMBEDDINGS_STEM = "all_embedding_text"
_UNIFIED_EMB_PATH = _DOMAIN_DIR / f"{_EMBEDDINGS_STEM}_embeddings.npy"
_CONFIG_PATH = _DOMAIN_DIR / "config.json"

_unified_embeddings: np.ndarray | None = None
_embedding_product_ids: list[str] | None = None


def _embedding_field_pairs_from_config(config_path: Path) -> list[tuple[str, str]]:
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


# ---------------------------------------------------------------------------
# Tool 1: filter_games
# ---------------------------------------------------------------------------

def _build_filter_games() -> DomainToolSpec:
    description = (
        "Filter the video game catalog using structured constraints. "
        "Supports filtering by genre, category/features, price, platform, "
        "language, developer, publisher, review count, recommendations, "
        "and release date. Returns matching games sorted by the chosen field."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "filter_games",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Filter by genre. Examples: Action, Adventure, RPG, "
                            "Strategy, Simulation, Indie, Casual, Free To Play"
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Filter by category/feature. Examples: Single-player, "
                            "Multi-player, Co-op, Online Co-op, Steam Achievements, "
                            "Full controller support, Steam Cloud, VR Only, "
                            "Remote Play Together, Steam Trading Cards, Steam Workshop"
                        ),
                    },
                    "price_max": {
                        "type": "number",
                        "description": "Maximum price in USD. Use 0 for free-to-play only.",
                    },
                    "supported_platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Require support for these platforms. Values: Windows, Mac, Linux.",
                    },
                    "supported_languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Require support for these interface languages. "
                            "Examples: English, French, German, Spanish - Spain, Russian, "
                            "Japanese, Simplified Chinese, Korean"
                        ),
                    },
                    "full_audio_languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Require full voice audio in these languages. "
                            "Examples: English, Japanese, French, German"
                        ),
                    },
                    "developer": {
                        "type": "string",
                        "description": "Filter by developer studio name (fuzzy matched).",
                    },
                    "publishers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by publisher name(s).",
                    },
                    "positive_min": {
                        "type": "integer",
                        "description": "Minimum number of positive reviews.",
                    },
                    "recommendations_min": {
                        "type": "integer",
                        "description": "Minimum number of user recommendations.",
                    },
                    "release_date_min": {
                        "type": "string",
                        "description": "Minimum release date (YYYY-MM-DD). Only games released on or after this date.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": [
                            "price", "positive", "recommendations",
                            "average_playtime_forever", "achievements", "release_date",
                        ],
                        "description": "Field to sort results by (default: positive).",
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (default: desc for popularity metrics, asc for price).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10, max 50).",
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
            msg = "No matching video games found."
            if corrections:
                msg += " Corrections: " + "; ".join(corrections)
            return msg

        sort_by = kwargs.get("sort_by", "positive")
        sort_order = kwargs.get("sort_order")
        limit = min(max(1, int(kwargs.get("limit", 10) or 10)), 50)

        if sort_by not in {"price", "positive", "recommendations",
                           "average_playtime_forever", "achievements", "release_date"}:
            sort_by = "positive"
        if sort_order not in {"asc", "desc"}:
            sort_order = "desc" if sort_by in {"positive", "recommendations",
                                                "average_playtime_forever", "achievements",
                                                "release_date"} else "asc"

        if sort_by in filtered.columns:
            filtered = filtered.sort_values(
                sort_by, ascending=(sort_order == "asc"), na_position="last"
            )

        top = filtered.head(limit)
        lines = [f"Found {total} matching video games."]
        if corrections:
            lines.append("Corrections: " + "; ".join(corrections))
        for _, row in top.iterrows():
            lines.append(_product_line(row))
        if total > limit:
            lines.append(f"Showing top {limit} of {total} results sorted by {sort_by} ({sort_order}).")
        return "\n".join(lines)

    return DomainToolSpec(name="filter_games", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# Tool 2: search_games
# ---------------------------------------------------------------------------

def _build_search_games() -> DomainToolSpec:
    _emb_index_blurb = _unified_embedding_index_description()
    description = (
        "Search the video game catalog using natural language. "
        "Supports two modes: 'keyword' (TF-IDF text matching) and 'semantic' "
        "(unified embedding similarity over long catalog text). "
        "Supports optional structured constraints to narrow results before searching. "
        "Search only considers rows that have usable source data (non-empty text in the "
        "indexed fields for keyword mode, or an entry in the embedding index for semantic mode); "
        "not every catalog row is searchable. "
        f"{_emb_index_blurb}"
    )

    TF_IDF_COLS = [
        "name", "short_description", "genres", "categories",
        "developer", "publishers",
    ]

    schema = {
        "type": "function",
        "function": {
            "name": "search_games",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["keyword", "semantic"],
                        "description": (
                            "Search mode. 'keyword' uses TF-IDF text matching (good for exact terms "
                            "like developer names, specific features). 'semantic' uses embedding similarity "
                            "against the unified all_embedding_text vector (covers the fields noted in "
                            "the tool description). Default: keyword."
                        ),
                    },
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: filter by genre before searching.",
                    },
                    "price_max": {
                        "type": "number",
                        "description": "Optional: max price filter.",
                    },
                    "supported_platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: require platform support (Windows, Mac, Linux).",
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

    def _row_blob(row: pd.Series, cols: list[str]) -> str:
        parts = []
        for col in cols:
            if col == "name":
                parts.append(str(row.name))
                continue
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
        limit = min(max(1, int(kwargs.get("limit", 10) or 10)), 30)

        constraints = _extract_constraints(kwargs)
        filtered = ctx.catalog
        corrections: list[str] = []
        if constraints:
            resolved, corrections = _fuzzy_resolve_constraints(constraints, ctx.config)
            try:
                filtered = GenericFilter.apply(filtered, resolved, ctx.config.constraints)
            except Exception:
                pass

        if len(filtered) == 0:
            return "No games matched the constraints."

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
                return f"No games matched the semantic query '{query}'."

            lines = [
                f"Semantic search results for '{query}' "
                f"(unified embedding {_EMBEDDINGS_STEM}, {len(sem_results)} results):"
            ]
            if corrections:
                lines.append("Corrections: " + "; ".join(corrections))

            for rank, (idx, score) in enumerate(sem_results, 1):
                row = filtered.loc[idx]
                name = str(idx)
                price = _fmt_price(row.get("price"))
                genres = _list_to_str(row.get("genres"), max_items=3)
                platforms = _list_to_str(row.get("supported_platforms"), max_items=3)
                review = _review_summary(
                    0 if _is_null(row.get("positive")) else row["positive"],
                    0 if _is_null(row.get("negative")) else row.get("negative", 0),
                )
                release = _fmt_date(row.get("release_date"))
                lines.append(
                    f"{rank}. {name} — {price}; {genres}; {review}; "
                    f"platforms: {platforms}; released: {release} [sim={score:.3f}]"
                )

            return "\n".join(lines)

        # Default: keyword / TF-IDF mode
        searchable = [c for c in TF_IDF_COLS if c in filtered.columns]
        query_tokens = _tokenize(query)
        if not query_tokens:
            return "Please provide a more specific search query."

        results = _tfidf_search(query, filtered, searchable, limit)

        if not results:
            return f"No games matched the query '{query}'."

        lines = [f"Search results for '{query}' ({len(results)} matches, showing top {min(limit, len(results))}):"]
        if corrections:
            lines.append("Corrections: " + "; ".join(corrections))

        for rank, item in enumerate(results, 1):
            row = filtered.loc[item["idx"]]
            name = str(item["idx"])
            price = _fmt_price(row.get("price"))
            genres = _list_to_str(row.get("genres"), max_items=3)
            platforms = _list_to_str(row.get("supported_platforms"), max_items=3)
            review = _review_summary(
                0 if _is_null(row.get("positive")) else row["positive"],
                0 if _is_null(row.get("negative")) else row.get("negative", 0),
            )
            release = _fmt_date(row.get("release_date"))
            lines.append(
                f"{rank}. {name} — {price}; {genres}; {review}; "
                f"platforms: {platforms}; released: {release}"
            )

        return "\n".join(lines)

    return DomainToolSpec(name="search_games", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# Tool 3: get_game_details
# ---------------------------------------------------------------------------

def _build_get_game_details() -> DomainToolSpec:
    description = (
        "Retrieve the full catalog record for a video game by its exact game title. "
        "Use the game name shown in search/filter results."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "get_game_details",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "game_name": {
                        "type": "string",
                        "description": "The exact game title as shown in search/filter results.",
                    },
                },
                "required": ["game_name"],
                "additionalProperties": False,
            },
        },
    }

    FIELD_ORDER = [
        "name", "developer",
        "genres", "categories",
        "price", "release_date",
        "supported_platforms", "supported_languages", "full_audio_languages",
        "positive", "recommendations",
        "achievements", "average_playtime_forever",
        "dlc_count", "estimated_owners", "peak_ccu",
        "publishers",
        "short_description",
    ]

    LABELS = {
        "name": "Game Title",
        "developer": "Developer",
        "genres": "Genres",
        "categories": "Store Features",
        "price": "Price",
        "release_date": "Release Date",
        "supported_platforms": "Platforms",
        "supported_languages": "Supported Languages",
        "full_audio_languages": "Full Audio Languages",
        "positive": "Positive Reviews",
        "recommendations": "User Recommendations",
        "achievements": "Achievements",
        "average_playtime_forever": "Average Playtime",
        "dlc_count": "DLC Count",
        "estimated_owners": "Estimated Owners",
        "peak_ccu": "Peak Concurrent Players",
        "publishers": "Publishers",
        "short_description": "Description",
    }

    PRICE_FIELDS = {"price"}
    DATE_FIELDS = {"release_date"}
    LIST_FIELDS = {"genres", "categories", "supported_platforms",
                   "supported_languages", "full_audio_languages", "publishers"}
    PLAYTIME_FIELDS = {"average_playtime_forever"}
    INT_FIELDS = {"positive", "recommendations", "achievements", "dlc_count",
                  "estimated_owners", "peak_ccu"}

    def _fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        game_name = kwargs.get("game_name")
        if not game_name:
            return "Missing required game_name."
        game_name = str(game_name).strip()

        row = ctx.get_product(game_name)
        if row is None:
            return f"Game not found: {game_name}"

        lines = [f"Game Title: {game_name}"]
        for col in FIELD_ORDER:
            if col == "name":
                continue
            if col not in row.index:
                continue
            v = row[col]
            if _is_null(v):
                continue

            label = LABELS.get(col, col.replace("_", " ").title())

            if col in PRICE_FIELDS:
                lines.append(f"{label}: {_fmt_price(v)}")
            elif col in DATE_FIELDS:
                lines.append(f"{label}: {_fmt_date(v)}")
            elif col in LIST_FIELDS:
                lines.append(f"{label}: {_list_to_str(v)}")
            elif col in PLAYTIME_FIELDS:
                lines.append(f"{label}: {_fmt_playtime(v)}")
            elif col in INT_FIELDS:
                lines.append(f"{label}: {_fmt_int(v)}")
            elif col == "short_description":
                s = str(v)
                if len(s) > 500:
                    s = s[:497] + "..."
                lines.append(f"{label}: {s}")
            else:
                lines.append(f"{label}: {v}")

        return "\n".join(lines)

    return DomainToolSpec(name="get_game_details", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# Tool 4: list_catalog_values
# ---------------------------------------------------------------------------

def _build_list_catalog_values() -> DomainToolSpec:
    description = (
        "Discover distinct values for a catalog field (e.g., genres, categories, "
        "developer, publishers, supported_languages) with counts. "
        "Supports fuzzy field name matching."
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
                            "Catalog field to inspect (e.g., genres, categories, "
                            "developer, publishers, supported_languages, "
                            "full_audio_languages, supported_platforms, price, "
                            "release_date, achievements, recommendations)."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional substring to filter values.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max values to return (default 30).",
                    },
                },
                "required": ["field_name"],
                "additionalProperties": False,
            },
        },
    }

    ALIASES = {
        "genre": "genres",
        "category": "categories",
        "feature": "categories",
        "features": "categories",
        "tag": "categories",
        "tags": "categories",
        "developers": "developer",
        "dev": "developer",
        "studio": "developer",
        "publisher": "publishers",
        "pub": "publishers",
        "language": "supported_languages",
        "languages": "supported_languages",
        "lang": "supported_languages",
        "audio": "full_audio_languages",
        "audio_language": "full_audio_languages",
        "audio_languages": "full_audio_languages",
        "voice": "full_audio_languages",
        "platform": "supported_platforms",
        "platforms": "supported_platforms",
        "os": "supported_platforms",
        "owner": "estimated_owners",
        "owners": "estimated_owners",
        "achievement": "achievements",
        "recommendation": "recommendations",
        "release": "release_date",
        "date": "release_date",
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
        limit = min(max(1, int(kwargs.get("limit", 30) or 30)), 200)

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
                if field in ("release_date",):
                    key = _fmt_date(v)
                else:
                    key = _normalize_text(v)
                if key:
                    counts[key] += 1
                    if key not in display_map:
                        if field in ("release_date",):
                            display_map[key] = _fmt_date(v)
                        else:
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

        if field not in set_valued_fields:
            col = ctx.catalog[field]
            if col.dtype in ("int64", "float64"):
                desc = col.dropna().describe()
                lines.append("")
                lines.append(f"  Min: {desc.get('min', 'N/A')}")
                lines.append(f"  Max: {desc.get('max', 'N/A')}")
                lines.append(f"  Mean: {desc.get('mean', 'N/A'):.2f}")
                lines.append(f"  Median: {desc.get('50%', 'N/A')}")

        return "\n".join(lines)

    return DomainToolSpec(name="list_catalog_values", description=description, fn=_fn, schema=schema)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_domain_specific_tools() -> list[DomainToolSpec]:
    """Build and return all domain-specific tools for the video game catalog.

    Called dynamically by the framework via ``load_domain_specific_tools()``.
    Does NOT include terminal tools (recommend_products, declare_infeasible)
    as those are added by the framework.
    """
    return [
        _build_filter_games(),
        _build_search_games(),
        _build_get_game_details(),
        _build_list_catalog_values(),
    ]
