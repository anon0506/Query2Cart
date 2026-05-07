"""Domain-specific tools for the cars recommendation benchmark.

Provides six tools tailored to the 11,914-car catalog:
  1. filter_cars      -- structured constraint filtering with sort/limit
  2. search_cars      -- TF-IDF text search with optional constraints
  3. get_car_details  -- full detail view for one car (by integer car_id)
  4. compare_cars     -- side-by-side comparison of 2-3 cars
  5. list_catalog_values  -- discover available values for any catalog field
  6. summarize_result_set -- aggregate stats for a set of car_ids

Entry point: ``build_domain_specific_tools()`` returns ``list[DomainToolSpec]``.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from difflib import SequenceMatcher
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
    "make",
    "model",
    "market_category",
    "vehicle_style",
    "vehicle_size",
    "engine_fuel_type",
    "driven_wheels",
]

_SORT_FIELDS = ["msrp", "engine_hp", "highway_mpg", "city_mpg", "year", "popularity"]


def _safe_int(val: Any) -> int | None:
    """Convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _fmt_price(val: Any) -> str:
    """Format a numeric price as $XX,XXX."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    try:
        return f"${int(val):,}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_hp(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    try:
        return f"{int(val)} HP"
    except (ValueError, TypeError):
        return str(val)


def _fmt_mpg(highway: Any, city: Any) -> str:
    parts: list[str] = []
    try:
        if highway is not None and not (isinstance(highway, float) and math.isnan(highway)):
            parts.append(f"{int(highway)} hwy")
    except (ValueError, TypeError):
        pass
    try:
        if city is not None and not (isinstance(city, float) and math.isnan(city)):
            parts.append(f"{int(city)} city")
    except (ValueError, TypeError):
        pass
    if not parts:
        return "N/A"
    return " / ".join(parts) + " MPG"


def _safe_val(row: pd.Series, col: str) -> Any:
    """Safely get a value from a Series, returning None for missing/NaN."""
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


def _market_cats(val: Any) -> list[str]:
    """Parse a market_category cell into a list of category strings."""
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        return [c.strip() for c in val.split(",") if c.strip()]
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(c).strip() for c in val if str(c).strip()]
    return [str(val)]


def _car_one_liner(row: pd.Series, car_id: int) -> str:
    """Format a single car as a compact one-line summary."""
    year = _safe_val(row, "year")
    make = _safe_val(row, "make") or "Unknown"
    model = _safe_val(row, "model") or ""
    price = _fmt_price(_safe_val(row, "msrp"))
    hp = _fmt_hp(_safe_val(row, "engine_hp"))
    mpg = _fmt_mpg(_safe_val(row, "highway_mpg"), _safe_val(row, "city_mpg"))
    drivetrain = _safe_val(row, "driven_wheels") or "N/A"
    transmission = _safe_val(row, "transmission_type") or "N/A"

    # Capitalize drivetrain nicely
    dw_map = {
        "front wheel drive": "FWD",
        "rear wheel drive": "RWD",
        "all wheel drive": "AWD",
        "four wheel drive": "4WD",
    }
    drivetrain_short = dw_map.get(str(drivetrain).lower(), str(drivetrain))

    trans_map = {
        "AUTOMATIC": "Automatic",
        "MANUAL": "Manual",
        "AUTOMATED_MANUAL": "Automated Manual",
        "DIRECT_DRIVE": "Direct Drive",
        "UNKNOWN": "Unknown",
    }
    trans_short = trans_map.get(str(transmission), str(transmission))

    year_str = str(int(year)) if year is not None else "????"
    return (
        f"{year_str} {make} {model} (car_id: {car_id}) "
        f"— {price}; {hp}; {mpg}; {drivetrain_short}; {trans_short}"
    )


def _get_car_row(ctx: DomainCatalogContext, car_id: int) -> pd.Series | None:
    """Look up a car by integer ID directly from the catalog index."""
    try:
        if car_id in ctx.catalog.index:
            return ctx.catalog.loc[car_id]
    except (KeyError, TypeError):
        pass
    return None


def _tokenize(text: str) -> list[str]:
    """Tokenize text for TF-IDF: lowercase, split on non-alphanum, drop short tokens."""
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t and len(t) >= 2]


def _fuzzy_field_match(user_field: str, valid_fields: list[str]) -> str | None:
    """Fuzzy-match a user-provided field name against valid catalog field names."""
    user_lower = user_field.lower().strip().replace(" ", "_")

    # Exact match
    for f in valid_fields:
        if user_lower == f.lower():
            return f

    # Common aliases
    aliases = {
        "brand": "make",
        "manufacturer": "make",
        "price": "msrp",
        "body_style": "vehicle_style",
        "body": "vehicle_style",
        "style": "vehicle_style",
        "size": "vehicle_size",
        "drivetrain": "driven_wheels",
        "drive": "driven_wheels",
        "transmission": "transmission_type",
        "fuel": "engine_fuel_type",
        "fuel_type": "engine_fuel_type",
        "hp": "engine_hp",
        "horsepower": "engine_hp",
        "cylinders": "engine_cylinders",
        "doors": "number_of_doors",
        "category": "market_category",
        "categories": "market_category",
        "hwy_mpg": "highway_mpg",
        "hwy": "highway_mpg",
        "city": "city_mpg",
        "mpg": "highway_mpg",
    }
    if user_lower in aliases:
        target = aliases[user_lower]
        if target in valid_fields:
            return target

    # Substring match
    for f in valid_fields:
        if user_lower in f.lower() or f.lower() in user_lower:
            return f

    # SequenceMatcher
    best_ratio = 0.0
    best_field = None
    for f in valid_fields:
        ratio = SequenceMatcher(None, user_lower, f.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_field = f
    if best_ratio >= 0.6 and best_field is not None:
        return best_field

    return None


# ---------------------------------------------------------------------------
# 1. filter_cars
# ---------------------------------------------------------------------------

def _build_filter_cars_tool() -> DomainToolSpec:
    """Build the filter_cars tool."""

    schema = {
        "type": "function",
        "function": {
            "name": "filter_cars",
            "description": (
                "Filter the car catalog using structured constraints. "
                "Returns cars matching ALL specified criteria. "
                "Only specify constraints the user has mentioned. "
                "Values are fuzzy-matched so natural phrasing works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "driven_wheels": {
                        "type": "string",
                        "description": "Drivetrain type",
                        "enum": [
                            "front wheel drive",
                            "rear wheel drive",
                            "all wheel drive",
                            "four wheel drive",
                        ],
                    },
                    "engine_cylinders_min": {
                        "type": "integer",
                        "description": "Minimum number of engine cylinders (e.g. 4, 6, 8)",
                    },
                    "engine_fuel_type": {
                        "type": "string",
                        "description": "Engine fuel type",
                        "enum": [
                            "regular unleaded",
                            "premium unleaded (required)",
                            "premium unleaded (recommended)",
                            "flex-fuel (unleaded/E85)",
                            "diesel",
                            "electric",
                        ],
                    },
                    "make": {
                        "type": "string",
                        "description": "Car make/brand (e.g. Toyota, BMW, Ford)",
                    },
                    "market_category_includes": {
                        "type": "string",
                        "description": (
                            "Market category the car must belong to "
                            "(e.g. Luxury, Performance, Crossover, Hybrid, Exotic)"
                        ),
                    },
                    "msrp_max": {
                        "type": "integer",
                        "description": "Maximum MSRP price in dollars",
                    },
                    "number_of_doors_min": {
                        "type": "integer",
                        "description": "Minimum number of doors (2 or 4)",
                    },
                    "transmission_type": {
                        "type": "string",
                        "description": "Transmission type",
                        "enum": ["AUTOMATIC", "MANUAL", "AUTOMATED_MANUAL", "DIRECT_DRIVE"],
                    },
                    "vehicle_size": {
                        "type": "string",
                        "description": "Vehicle size category",
                        "enum": ["Compact", "Midsize", "Large"],
                    },
                    "vehicle_style": {
                        "type": "string",
                        "description": "Vehicle body style (e.g. Sedan, 4dr SUV, Coupe, Convertible)",
                        "enum": [
                            "Sedan",
                            "4dr SUV",
                            "Coupe",
                            "Convertible",
                            "4dr Hatchback",
                            "Crew Cab Pickup",
                            "Extended Cab Pickup",
                            "Wagon",
                            "2dr Hatchback",
                            "Passenger Minivan",
                            "Regular Cab Pickup",
                            "2dr SUV",
                            "Passenger Van",
                            "Cargo Van",
                            "Cargo Minivan",
                        ],
                    },
                    "year_min": {
                        "type": "integer",
                        "description": "Minimum model year (e.g. 2010, 2015)",
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

    param_descriptions = {
        "driven_wheels": "Drivetrain type",
        "engine_cylinders_min": "Minimum number of engine cylinders",
        "engine_fuel_type": "Engine fuel type",
        "make": "Car make/brand",
        "market_category_includes": "Market category the car must belong to",
        "msrp_max": "Maximum MSRP price in dollars",
        "number_of_doors_min": "Minimum number of doors",
        "transmission_type": "Transmission type",
        "vehicle_size": "Vehicle size category",
        "vehicle_style": "Vehicle body style",
        "year_min": "Minimum model year",
        "sort_by": "Field to sort results by",
        "sort_order": "Sort direction (asc or desc)",
        "limit": "Maximum number of results to return",
    }

    def _filter_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        sort_by = kwargs.pop("sort_by", None)
        sort_order = kwargs.pop("sort_order", "asc")
        limit = min(int(kwargs.pop("limit", 10)), 50)

        raw_constraints = {k: v for k, v in kwargs.items() if v is not None}

        if not raw_constraints:
            sample = ctx.catalog.head(limit)
            lines = [f"Showing first {len(sample)} cars (no filters applied)."]
            for idx, row in sample.iterrows():
                lines.append(f"- {_car_one_liner(row, int(idx))}")
            return "\n".join(lines)

        # Wrap single make value in a list for the in_set operator
        if "make" in raw_constraints and isinstance(raw_constraints["make"], str):
            raw_constraints["make"] = [raw_constraints["make"]]

        constraints, corrections = _fuzzy_resolve_constraints(raw_constraints, ctx.config)

        try:
            filtered = GenericFilter.apply(
                ctx.catalog, constraints, ctx.config.constraints
            )
        except Exception as exc:
            logger.warning("GenericFilter.apply failed: %s. Falling back to manual filter.", exc)
            filtered = _manual_filter(ctx.catalog, constraints)

        # Sort
        if sort_by and sort_by in filtered.columns and len(filtered) > 0:
            ascending = sort_order.lower() != "desc"
            filtered = filtered.sort_values(
                by=sort_by, ascending=ascending, na_position="last"
            )

        total = len(filtered)
        display = filtered.head(limit)

        lines: list[str] = []
        if corrections:
            lines.append("Note: " + "; ".join(corrections))
        if total == 0:
            lines.append("No cars matched all specified constraints. Try relaxing some filters.")
            return "\n".join(lines)

        lines.append(f"Found {total} matching cars{f' (showing top {limit})' if total > limit else ''}.")
        for idx, row in display.iterrows():
            lines.append(f"- {_car_one_liner(row, int(idx))}")
        return "\n".join(lines)

    return DomainToolSpec(
        name="filter_cars",
        description=schema["function"]["description"],
        fn=_filter_fn,
        schema=schema,
        param_descriptions=param_descriptions,
    )


def _manual_filter(catalog: pd.DataFrame, constraints: dict[str, Any]) -> pd.DataFrame:
    """Fallback manual filtering when GenericFilter fails."""
    mask = pd.Series(True, index=catalog.index)

    op_map = {
        "driven_wheels": ("driven_wheels", "eq"),
        "engine_cylinders_min": ("engine_cylinders", "gte"),
        "engine_fuel_type": ("engine_fuel_type", "eq"),
        "make": ("make", "in_set"),
        "market_category_includes": ("market_category", "contains_any"),
        "msrp_max": ("msrp", "lte"),
        "number_of_doors_min": ("number_of_doors", "gte"),
        "transmission_type": ("transmission_type", "eq"),
        "vehicle_size": ("vehicle_size", "eq"),
        "vehicle_style": ("vehicle_style", "in_set"),
        "year_min": ("year", "gte"),
    }

    for cname, val in constraints.items():
        if cname not in op_map:
            continue
        col_name, op = op_map[cname]
        if col_name not in catalog.columns:
            continue
        col = catalog[col_name]

        if op == "eq":
            mask &= col.str.lower().fillna("") == str(val).lower()
        elif op == "gte":
            mask &= col.fillna(-np.inf) >= float(val)
        elif op == "lte":
            mask &= col.fillna(np.inf) <= float(val)
        elif op == "in_set":
            if isinstance(val, str):
                val = [val]
            lowered = [str(v).lower() for v in val]
            mask &= col.str.lower().fillna("").isin(lowered)
        elif op == "contains_any":
            targets = [val] if isinstance(val, str) else list(val)
            targets_lower = [str(t).lower() for t in targets]

            def _check_contains_any(cell: Any) -> bool:
                cats = _market_cats(cell)
                cats_lower = [c.lower() for c in cats]
                return any(t in cats_lower for t in targets_lower)

            mask &= col.apply(_check_contains_any)

    return catalog[mask]


# ---------------------------------------------------------------------------
# 2. search_cars
# ---------------------------------------------------------------------------

def _build_search_cars_tool() -> DomainToolSpec:
    """Build the search_cars tool using TF-IDF across text fields."""

    schema = {
        "type": "function",
        "function": {
            "name": "search_cars",
            "description": (
                "Search the car catalog using a natural language query. "
                "Searches across make, model, market category, body style, "
                "vehicle size, fuel type, and drivetrain. "
                "Results are ranked by relevance and boosted by popularity. "
                "Supports optional structured constraints to narrow results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'luxury SUV', 'Toyota hybrid sedan')",
                    },
                    "msrp_max": {
                        "type": "integer",
                        "description": "Optional: maximum MSRP to filter results",
                    },
                    "year_min": {
                        "type": "integer",
                        "description": "Optional: minimum model year to filter results",
                    },
                    "driven_wheels": {
                        "type": "string",
                        "description": "Optional: drivetrain filter",
                    },
                    "transmission_type": {
                        "type": "string",
                        "description": "Optional: transmission type filter",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 10, max 30)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    def _search_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        query = str(kwargs.get("query", ""))
        top_k = min(max(1, int(kwargs.get("top_k", 10))), 30)

        # Apply optional structured constraints first
        constraint_keys = {"msrp_max", "year_min", "driven_wheels", "transmission_type"}
        raw_constraints = {
            k: v for k, v in kwargs.items()
            if k in constraint_keys and v is not None
        }

        if raw_constraints:
            # Wrap make if needed (not a constraint here but be safe)
            constraints, corrections = _fuzzy_resolve_constraints(raw_constraints, ctx.config)
            try:
                catalog = GenericFilter.apply(
                    ctx.catalog, constraints, ctx.config.constraints
                )
            except Exception:
                catalog = ctx.catalog
        else:
            catalog = ctx.catalog
            corrections = []

        if len(catalog) == 0:
            return "No cars match the specified constraints."

        query_tokens = _tokenize(query)
        if not query_tokens:
            sample = catalog.head(top_k)
            lines = [f"Showing {len(sample)} cars (empty query)."]
            for rank, (idx, row) in enumerate(sample.iterrows(), 1):
                lines.append(f"  {rank}. {_car_one_liner(row, int(idx))}")
            return "\n".join(lines)

        # Build document text for each row
        available_fields = [f for f in _SEARCH_FIELDS if f in catalog.columns]

        def _row_text(row: pd.Series) -> str:
            parts: list[str] = []
            for col in available_fields:
                val = _safe_val(row, col)
                if val is None:
                    continue
                if col == "market_category":
                    cats = _market_cats(val)
                    parts.extend(cats)
                else:
                    parts.append(str(val))
            return " ".join(parts)

        # Compute IDF from the catalog
        n_docs = len(catalog)
        doc_freq: Counter = Counter()
        row_tokens_map: dict[Any, set[str]] = {}

        for idx, row in catalog.iterrows():
            text = _row_text(row)
            tokens = set(_tokenize(text))
            row_tokens_map[idx] = tokens
            for qt in query_tokens:
                if qt in tokens:
                    doc_freq[qt] += 1
                else:
                    # Check substring match for IDF
                    for rt in tokens:
                        if qt in rt or rt in qt:
                            doc_freq[qt] += 1
                            break

        idf: dict[str, float] = {}
        for qt in query_tokens:
            df = doc_freq.get(qt, 0)
            idf[qt] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # Compute popularity normalization for boosting
        pop_col = "popularity"
        has_popularity = pop_col in catalog.columns
        if has_popularity:
            pop_max = catalog[pop_col].max()
            pop_min = catalog[pop_col].min()
            pop_range = max(pop_max - pop_min, 1)

        # Score each document
        scored: list[tuple[float, Any]] = []
        for idx, tokens in row_tokens_map.items():
            score = 0.0
            for qt in query_tokens:
                if qt in tokens:
                    score += idf.get(qt, 1.0)
                else:
                    for rt in tokens:
                        if qt in rt or rt in qt:
                            score += idf.get(qt, 1.0) * 0.5
                            break

            if score > 0:
                # Boost by popularity (small factor to avoid overwhelming relevance)
                if has_popularity:
                    pop_val = _safe_val(catalog.loc[idx], pop_col)
                    if pop_val is not None:
                        pop_norm = (float(pop_val) - pop_min) / pop_range
                        score *= (1.0 + 0.2 * pop_norm)
                scored.append((score, idx))

        scored.sort(key=lambda x: -x[0])
        top_ids = [s[1] for s in scored[:top_k]]

        if not top_ids:
            return "No cars matched the search query."

        lines: list[str] = []
        if corrections:
            lines.append("Note: " + "; ".join(corrections))
        lines.append(f"Found {len(scored)} relevant cars (showing top {min(top_k, len(scored))}).")
        for rank, idx in enumerate(top_ids, 1):
            row = catalog.loc[idx]
            lines.append(f"  {rank}. {_car_one_liner(row, int(idx))}")
        return "\n".join(lines)

    return DomainToolSpec(
        name="search_cars",
        description=schema["function"]["description"],
        fn=_search_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# 3. get_car_details
# ---------------------------------------------------------------------------

def _build_get_car_details_tool() -> DomainToolSpec:
    """Build the get_car_details tool."""

    schema = {
        "type": "function",
        "function": {
            "name": "get_car_details",
            "description": (
                "Get full details for a specific car by its car_id. "
                "Returns all available information including price, engine, "
                "transmission, MPG, size, style, and market categories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {
                        "type": "integer",
                        "description": "The car ID (integer)",
                    },
                },
                "required": ["car_id"],
                "additionalProperties": False,
            },
        },
    }

    _DISPLAY_ORDER = [
        ("car_id", "Car ID"),
        ("year", "Model Year"),
        ("make", "Make"),
        ("model", "Model"),
        ("msrp", "MSRP"),
        ("engine_hp", "Horsepower"),
        ("engine_cylinders", "Engine Cylinders"),
        ("engine_fuel_type", "Fuel Type"),
        ("transmission_type", "Transmission"),
        ("driven_wheels", "Drivetrain"),
        ("number_of_doors", "Number of Doors"),
        ("vehicle_size", "Vehicle Size"),
        ("vehicle_style", "Body Style"),
        ("market_category", "Market Category"),
        ("highway_mpg", "Highway MPG"),
        ("city_mpg", "City MPG"),
        ("popularity", "Popularity Score"),
    ]

    def _detail_fn(ctx: DomainCatalogContext, car_id: Any = None, **kwargs: Any) -> str:
        if car_id is None:
            car_id = kwargs.get("car_id")
        if car_id is None:
            return "Error: car_id is required."

        cid = _safe_int(car_id)
        if cid is None:
            return f"Error: invalid car_id '{car_id}'. Must be an integer."

        row = _get_car_row(ctx, cid)
        if row is None:
            return f"Car with car_id {cid} not found in the catalog."

        lines: list[str] = []
        year = _safe_val(row, "year")
        make = _safe_val(row, "make") or "Unknown"
        model = _safe_val(row, "model") or ""
        year_str = str(int(year)) if year is not None else "Unknown"
        lines.append(f"=== {year_str} {make} {model} ===")
        lines.append("")

        for col, label in _DISPLAY_ORDER:
            if col == "car_id":
                lines.append(f"  {label}: {cid}")
                continue

            val = _safe_val(row, col)
            if val is None:
                lines.append(f"  {label}: N/A")
                continue

            if col == "msrp":
                lines.append(f"  {label}: {_fmt_price(val)}")
            elif col == "engine_hp":
                lines.append(f"  {label}: {_fmt_hp(val)}")
            elif col == "engine_cylinders":
                cyl = int(val) if val == val else "N/A"
                lines.append(f"  {label}: {cyl}")
            elif col == "number_of_doors":
                doors = int(val) if val == val else "N/A"
                lines.append(f"  {label}: {doors}")
            elif col == "market_category":
                cats = _market_cats(val)
                lines.append(f"  {label}: {', '.join(cats) if cats else 'None'}")
            elif col == "highway_mpg":
                city = _safe_val(row, "city_mpg")
                lines.append(f"  Fuel Economy: {_fmt_mpg(val, city)}")
            elif col == "city_mpg":
                # Already covered in highway_mpg line
                continue
            elif col in ("year", "popularity"):
                lines.append(f"  {label}: {int(val)}")
            else:
                lines.append(f"  {label}: {val}")

        return "\n".join(lines)

    return DomainToolSpec(
        name="get_car_details",
        description=schema["function"]["description"],
        fn=_detail_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# 4. compare_cars
# ---------------------------------------------------------------------------

def _build_compare_cars_tool() -> DomainToolSpec:
    """Build the compare_cars tool."""

    schema = {
        "type": "function",
        "function": {
            "name": "compare_cars",
            "description": (
                "Compare 2 to 3 cars side by side on price, engine, transmission, "
                "drivetrain, MPG, size, style, and market categories. "
                "Highlights key differences like cheapest, most powerful, and best MPG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 3,
                        "description": "List of 2-3 car_id values (integers) to compare",
                    },
                },
                "required": ["product_ids"],
                "additionalProperties": False,
            },
        },
    }

    _COMPARE_ATTRS = [
        ("msrp", "MSRP", "price"),
        ("year", "Model Year", "int"),
        ("engine_hp", "Horsepower", "hp"),
        ("engine_cylinders", "Cylinders", "int"),
        ("engine_fuel_type", "Fuel Type", "str"),
        ("transmission_type", "Transmission", "str"),
        ("driven_wheels", "Drivetrain", "str"),
        ("highway_mpg", "Highway MPG", "int"),
        ("city_mpg", "City MPG", "int"),
        ("vehicle_size", "Vehicle Size", "str"),
        ("vehicle_style", "Body Style", "str"),
        ("market_category", "Market Category", "cats"),
        ("number_of_doors", "Doors", "int"),
        ("popularity", "Popularity", "int"),
    ]

    def _compare_fn(ctx: DomainCatalogContext, product_ids: list | None = None, **kwargs: Any) -> str:
        if product_ids is None:
            product_ids = kwargs.get("product_ids", [])

        if not product_ids or len(product_ids) < 2:
            return "Error: provide at least 2 car_ids to compare."
        if len(product_ids) > 3:
            product_ids = product_ids[:3]

        cars: list[tuple[int, pd.Series]] = []
        not_found: list[str] = []

        for pid in product_ids:
            cid = _safe_int(pid)
            if cid is None:
                not_found.append(str(pid))
                continue
            row = _get_car_row(ctx, cid)
            if row is None:
                not_found.append(str(pid))
            else:
                cars.append((cid, row))

        if not_found:
            msg = f"Car(s) not found: {', '.join(not_found)}. "
            if len(cars) < 2:
                return msg + "Need at least 2 valid cars to compare."

        # Build headers
        headers: list[str] = []
        for cid, row in cars:
            year = _safe_val(row, "year")
            make = _safe_val(row, "make") or "?"
            model = _safe_val(row, "model") or ""
            year_str = str(int(year)) if year is not None else "????"
            headers.append(f"{year_str} {make} {model} (#{cid})")

        lines: list[str] = ["=== Car Comparison ===", ""]

        # Name header line
        col_width = 28
        label_width = 18
        header_line = " " * label_width
        for h in headers:
            header_line += f"  {h:<{col_width}}"
        lines.append(header_line)
        lines.append("-" * len(header_line))

        for col, label, fmt in _COMPARE_ATTRS:
            row_line = f"{label:<{label_width}}"
            for cid, row in cars:
                val = _safe_val(row, col)
                if val is None:
                    cell = "N/A"
                elif fmt == "price":
                    cell = _fmt_price(val)
                elif fmt == "hp":
                    cell = _fmt_hp(val)
                elif fmt == "int":
                    try:
                        cell = str(int(val))
                    except (ValueError, TypeError):
                        cell = str(val)
                elif fmt == "cats":
                    cats = _market_cats(val)
                    cell = ", ".join(cats) if cats else "None"
                else:
                    cell = str(val)
                row_line += f"  {cell:<{col_width}}"
            lines.append(row_line)

        # Key differences / highlights
        lines.append("")
        lines.append("--- Key Differences ---")

        # Cheapest
        valid_prices = [
            (cid, _safe_val(row, "msrp"))
            for cid, row in cars
            if _safe_val(row, "msrp") is not None
        ]
        if valid_prices:
            cheapest = min(valid_prices, key=lambda x: x[1])
            cheapest_row = next(r for c, r in cars if c == cheapest[0])
            make = _safe_val(cheapest_row, "make") or "?"
            model = _safe_val(cheapest_row, "model") or ""
            lines.append(
                f"  Cheapest: {make} {model} (#{cheapest[0]}) at {_fmt_price(cheapest[1])}"
            )

        # Most powerful
        valid_hp = [
            (cid, _safe_val(row, "engine_hp"))
            for cid, row in cars
            if _safe_val(row, "engine_hp") is not None
        ]
        if valid_hp:
            strongest = max(valid_hp, key=lambda x: x[1])
            strongest_row = next(r for c, r in cars if c == strongest[0])
            make = _safe_val(strongest_row, "make") or "?"
            model = _safe_val(strongest_row, "model") or ""
            lines.append(
                f"  Most Powerful: {make} {model} (#{strongest[0]}) with {_fmt_hp(strongest[1])}"
            )

        # Best highway MPG
        valid_mpg = [
            (cid, _safe_val(row, "highway_mpg"))
            for cid, row in cars
            if _safe_val(row, "highway_mpg") is not None
        ]
        if valid_mpg:
            best_mpg = max(valid_mpg, key=lambda x: x[1])
            best_mpg_row = next(r for c, r in cars if c == best_mpg[0])
            make = _safe_val(best_mpg_row, "make") or "?"
            model = _safe_val(best_mpg_row, "model") or ""
            city = _safe_val(best_mpg_row, "city_mpg")
            lines.append(
                f"  Best MPG: {make} {model} (#{best_mpg[0]}) at {_fmt_mpg(best_mpg[1], city)}"
            )

        return "\n".join(lines)

    return DomainToolSpec(
        name="compare_cars",
        description=schema["function"]["description"],
        fn=_compare_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# 5. list_catalog_values
# ---------------------------------------------------------------------------

def _build_list_catalog_values_tool() -> DomainToolSpec:
    """Build the list_catalog_values tool."""

    _LISTABLE_FIELDS = [
        "make",
        "model",
        "year",
        "engine_fuel_type",
        "engine_cylinders",
        "transmission_type",
        "driven_wheels",
        "number_of_doors",
        "market_category",
        "vehicle_size",
        "vehicle_style",
        "highway_mpg",
        "city_mpg",
        "engine_hp",
        "msrp",
        "popularity",
    ]

    schema = {
        "type": "function",
        "function": {
            "name": "list_catalog_values",
            "description": (
                "Discover the available values for any catalog field. "
                "Returns unique values with their counts, sorted by frequency. "
                "Supports fuzzy field name matching (e.g. 'brand' matches 'make'). "
                f"Available fields: {', '.join(_LISTABLE_FIELDS)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "description": (
                            "The catalog field to list values for "
                            "(e.g. 'make', 'vehicle_style', 'fuel_type')"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of values to return (default 30)",
                    },
                },
                "required": ["field_name"],
                "additionalProperties": False,
            },
        },
    }

    def _list_fn(ctx: DomainCatalogContext, field_name: str = "", top_k: int = 30, **kwargs: Any) -> str:
        if not field_name:
            field_name = kwargs.get("field_name", "")
        top_k = min(max(1, int(kwargs.get("top_k", top_k))), 200)

        valid_fields = [f for f in _LISTABLE_FIELDS if f in ctx.catalog.columns]

        resolved = _fuzzy_field_match(field_name, valid_fields)
        if resolved is None:
            return (
                f"Unknown field '{field_name}'. "
                f"Available fields: {', '.join(valid_fields)}"
            )

        col = ctx.catalog[resolved]
        non_null_count = col.notna().sum()
        null_count = col.isna().sum()

        # Handle set-valued field (market_category)
        if resolved == "market_category":
            counter: Counter = Counter()
            for val in col.dropna():
                for cat in _market_cats(val):
                    counter[cat] += 1

            lines = [
                f"Field: {resolved} (set-valued, comma-separated)",
                f"Non-null rows: {non_null_count:,} / {len(ctx.catalog):,} ({null_count:,} null)",
                f"Unique categories: {len(counter)}",
                "",
                "Values (by frequency):",
            ]
            for val, count in counter.most_common(top_k):
                lines.append(f"  {val}: {count:,}")
            return "\n".join(lines)

        # Numeric fields: show range + distribution summary
        if pd.api.types.is_numeric_dtype(col):
            desc = col.describe()
            unique_count = col.nunique()

            lines = [
                f"Field: {resolved} (numeric)",
                f"Non-null rows: {non_null_count:,} / {len(ctx.catalog):,} ({null_count:,} null)",
                f"Unique values: {unique_count:,}",
                "",
                f"Range: {desc.get('min', 'N/A')} to {desc.get('max', 'N/A')}",
                f"Mean: {desc.get('mean', 0):.1f}",
                f"Median: {desc.get('50%', 0):.1f}",
                f"Std Dev: {desc.get('std', 0):.1f}",
            ]

            # If few unique values, show them all with counts
            if unique_count <= 30:
                lines.append("")
                lines.append("Values (by frequency):")
                value_counts = col.dropna().value_counts().head(top_k)
                for val, count in value_counts.items():
                    try:
                        val_display = str(int(val)) if float(val) == int(val) else str(val)
                    except (ValueError, TypeError):
                        val_display = str(val)
                    lines.append(f"  {val_display}: {count:,}")

            return "\n".join(lines)

        # Categorical fields
        value_counts = col.dropna().value_counts().head(top_k)
        unique_count = col.nunique()

        lines = [
            f"Field: {resolved} (categorical)",
            f"Non-null rows: {non_null_count:,} / {len(ctx.catalog):,} ({null_count:,} null)",
            f"Unique values: {unique_count:,}",
            "",
            "Values (by frequency):",
        ]
        for val, count in value_counts.items():
            lines.append(f"  {val}: {count:,}")

        return "\n".join(lines)

    return DomainToolSpec(
        name="list_catalog_values",
        description=schema["function"]["description"],
        fn=_list_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# 6. summarize_result_set
# ---------------------------------------------------------------------------

def _build_summarize_result_set_tool() -> DomainToolSpec:
    """Build the summarize_result_set tool."""

    schema = {
        "type": "function",
        "function": {
            "name": "summarize_result_set",
            "description": (
                "Summarize a set of cars by their car_ids. "
                "Shows price range, top makes, body style mix, MPG range, "
                "engine stats, year spread, and standout vehicles "
                "(cheapest, most powerful, best MPG)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of car_id values (integers) to summarize",
                    },
                },
                "required": ["product_ids"],
                "additionalProperties": False,
            },
        },
    }

    def _summarize_fn(ctx: DomainCatalogContext, product_ids: list | None = None, **kwargs: Any) -> str:
        if product_ids is None:
            product_ids = kwargs.get("product_ids", [])

        if not product_ids:
            return "Error: provide at least one car_id to summarize."

        # Resolve IDs and collect rows
        rows: list[tuple[int, pd.Series]] = []
        not_found: list[str] = []

        for pid in product_ids:
            cid = _safe_int(pid)
            if cid is None:
                not_found.append(str(pid))
                continue
            row = _get_car_row(ctx, cid)
            if row is None:
                not_found.append(str(pid))
            else:
                rows.append((cid, row))

        if not rows:
            return f"None of the provided car_ids were found: {', '.join(not_found)}"

        lines: list[str] = [f"=== Summary of {len(rows)} Cars ===", ""]

        if not_found:
            lines.append(f"Note: {len(not_found)} car_id(s) not found: {', '.join(not_found[:10])}")
            lines.append("")

        # Price range
        prices = [
            (_safe_val(r, "msrp"), cid) for cid, r in rows if _safe_val(r, "msrp") is not None
        ]
        if prices:
            min_price = min(prices, key=lambda x: x[0])
            max_price = max(prices, key=lambda x: x[0])
            avg_price = sum(p for p, _ in prices) / len(prices)
            lines.append(
                f"Price Range: {_fmt_price(min_price[0])} - {_fmt_price(max_price[0])} "
                f"(avg {_fmt_price(avg_price)})"
            )

        # Year spread
        years = [_safe_val(r, "year") for _, r in rows if _safe_val(r, "year") is not None]
        if years:
            lines.append(f"Year Range: {int(min(years))} - {int(max(years))}")

        lines.append("")

        # Top makes
        make_counts: Counter = Counter()
        for _, r in rows:
            m = _safe_val(r, "make")
            if m:
                make_counts[str(m)] += 1
        if make_counts:
            lines.append("Makes:")
            for make, count in make_counts.most_common(8):
                lines.append(f"  {make}: {count}")

        # Body style mix
        style_counts: Counter = Counter()
        for _, r in rows:
            s = _safe_val(r, "vehicle_style")
            if s:
                style_counts[str(s)] += 1
        if style_counts:
            lines.append("")
            lines.append("Body Styles:")
            for style, count in style_counts.most_common(8):
                lines.append(f"  {style}: {count}")

        # Drivetrain mix
        drive_counts: Counter = Counter()
        for _, r in rows:
            d = _safe_val(r, "driven_wheels")
            if d:
                drive_counts[str(d)] += 1
        if drive_counts:
            lines.append("")
            lines.append("Drivetrains:")
            for dw, count in drive_counts.most_common():
                lines.append(f"  {dw}: {count}")

        # Engine stats
        hp_vals = [
            _safe_val(r, "engine_hp") for _, r in rows if _safe_val(r, "engine_hp") is not None
        ]
        if hp_vals:
            lines.append("")
            lines.append(
                f"Horsepower: {int(min(hp_vals))} - {int(max(hp_vals))} HP "
                f"(avg {int(sum(hp_vals) / len(hp_vals))} HP)"
            )

        cyl_counts: Counter = Counter()
        for _, r in rows:
            c = _safe_val(r, "engine_cylinders")
            if c is not None:
                cyl_counts[int(c)] += 1
        if cyl_counts:
            cyl_str = ", ".join(
                f"{cyl}-cyl: {cnt}" for cyl, cnt in sorted(cyl_counts.items())
            )
            lines.append(f"Cylinders: {cyl_str}")

        fuel_counts: Counter = Counter()
        for _, r in rows:
            f = _safe_val(r, "engine_fuel_type")
            if f:
                fuel_counts[str(f)] += 1
        if fuel_counts:
            lines.append("Fuel Types: " + ", ".join(
                f"{ft} ({c})" for ft, c in fuel_counts.most_common()
            ))

        # MPG range
        hwy_vals = [
            _safe_val(r, "highway_mpg") for _, r in rows if _safe_val(r, "highway_mpg") is not None
        ]
        city_vals = [
            _safe_val(r, "city_mpg") for _, r in rows if _safe_val(r, "city_mpg") is not None
        ]
        if hwy_vals or city_vals:
            lines.append("")
            if hwy_vals:
                lines.append(
                    f"Highway MPG: {int(min(hwy_vals))} - {int(max(hwy_vals))} "
                    f"(avg {int(sum(hwy_vals) / len(hwy_vals))})"
                )
            if city_vals:
                lines.append(
                    f"City MPG: {int(min(city_vals))} - {int(max(city_vals))} "
                    f"(avg {int(sum(city_vals) / len(city_vals))})"
                )

        # Market categories
        cat_counts: Counter = Counter()
        for _, r in rows:
            mc = _safe_val(r, "market_category")
            for cat in _market_cats(mc):
                cat_counts[cat] += 1
        if cat_counts:
            lines.append("")
            lines.append("Market Categories:")
            for cat, count in cat_counts.most_common(8):
                lines.append(f"  {cat}: {count}")

        # Standout vehicles
        lines.append("")
        lines.append("--- Standout Vehicles ---")

        if prices:
            cheapest_price, cheapest_id = min(prices, key=lambda x: x[0])
            cheapest_row = next(r for c, r in rows if c == cheapest_id)
            make = _safe_val(cheapest_row, "make") or "?"
            model = _safe_val(cheapest_row, "model") or ""
            year = _safe_val(cheapest_row, "year")
            year_str = str(int(year)) if year is not None else "????"
            lines.append(
                f"  Cheapest: {year_str} {make} {model} (#{cheapest_id}) "
                f"at {_fmt_price(cheapest_price)}"
            )

        hp_with_ids = [
            (_safe_val(r, "engine_hp"), cid)
            for cid, r in rows
            if _safe_val(r, "engine_hp") is not None
        ]
        if hp_with_ids:
            max_hp, max_hp_id = max(hp_with_ids, key=lambda x: x[0])
            max_hp_row = next(r for c, r in rows if c == max_hp_id)
            make = _safe_val(max_hp_row, "make") or "?"
            model = _safe_val(max_hp_row, "model") or ""
            year = _safe_val(max_hp_row, "year")
            year_str = str(int(year)) if year is not None else "????"
            lines.append(
                f"  Most Powerful: {year_str} {make} {model} (#{max_hp_id}) "
                f"with {_fmt_hp(max_hp)}"
            )

        hwy_with_ids = [
            (_safe_val(r, "highway_mpg"), cid)
            for cid, r in rows
            if _safe_val(r, "highway_mpg") is not None
        ]
        if hwy_with_ids:
            best_hwy, best_hwy_id = max(hwy_with_ids, key=lambda x: x[0])
            best_mpg_row = next(r for c, r in rows if c == best_hwy_id)
            make = _safe_val(best_mpg_row, "make") or "?"
            model = _safe_val(best_mpg_row, "model") or ""
            year = _safe_val(best_mpg_row, "year")
            year_str = str(int(year)) if year is not None else "????"
            city_mpg = _safe_val(best_mpg_row, "city_mpg")
            lines.append(
                f"  Best MPG: {year_str} {make} {model} (#{best_hwy_id}) "
                f"at {_fmt_mpg(best_hwy, city_mpg)}"
            )

        return "\n".join(lines)

    return DomainToolSpec(
        name="summarize_result_set",
        description=schema["function"]["description"],
        fn=_summarize_fn,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_domain_specific_tools() -> list[DomainToolSpec]:
    """Build and return all domain-specific tools for the cars domain.

    Called dynamically by the framework via ``load_domain_specific_tools()``.
    Does NOT include terminal tools (recommend_products, declare_infeasible)
    -- the framework adds those automatically.
    """
    return [
        _build_filter_cars_tool(),
        _build_search_cars_tool(),
        _build_get_car_details_tool(),
        _build_compare_cars_tool(),
        _build_list_catalog_values_tool(),
        _build_summarize_result_set_tool(),
    ]
