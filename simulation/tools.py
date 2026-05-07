"""Domain-agnostic catalog tools generated from DomainConfig.

Auto-generates filter, search, and detail tools from a DomainConfig's
constraint specs and attribute specs.  No hardcoded column names.

Usage:
    from shared.config import DomainConfig
    from simulation.tools import build_domain_tools, DomainCatalogContext

    config = DomainConfig.load("domain/food/config.json")
    ctx = DomainCatalogContext(catalog, config)
    tools = build_domain_tools(config)
"""

from __future__ import annotations

import importlib.util
import json as _json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from shared.config import AttrType, ConstraintOp, ConstraintSpec, DomainConfig
from shared.filter import GenericFilter

logger = logging.getLogger(__name__)


@dataclass
class DomainCatalogContext:
    """Typed context passed to domain-agnostic tool invocations."""

    catalog: pd.DataFrame
    config: DomainConfig

    def _coerce_id(self, product_id: Any) -> Any:
        """Cast *product_id* to the catalog index dtype so lookups match."""
        idx_dtype = self.catalog.index.dtype
        try:
            if pd.api.types.is_integer_dtype(idx_dtype):
                return int(product_id)
            if pd.api.types.is_float_dtype(idx_dtype):
                return float(product_id)
        except (ValueError, TypeError):
            pass
        return product_id

    def get_product(self, product_id: Any) -> pd.Series | None:
        product_id = self._coerce_id(product_id)
        if product_id not in self.catalog.index:
            return None
        return self.catalog.loc[product_id]


# -- ToolSpec ----------------------------

@dataclass
class DomainToolSpec:
    name: str
    description: str
    fn: Callable[..., str]
    schema: dict[str, Any]
    param_descriptions: dict[str, str] = field(default_factory=dict)

    def invoke(self, ctx: DomainCatalogContext, **kwargs: Any) -> str:
        return self.fn(ctx, **kwargs)


# -- Type mapping --------------------------------------------------------------

_VALUE_TYPE_TO_JSON: dict[str, str] = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list[str]": "string",
}

_OP_HINT: dict[ConstraintOp, str] = {
    ConstraintOp.LTE: "Maximum",
    ConstraintOp.GTE: "Minimum",
    ConstraintOp.EQ: "Exact match",
    ConstraintOp.NEQ: "Exclude",
    ConstraintOp.IN_SET: "One of",
    ConstraintOp.CONTAINS: "Must include",
    ConstraintOp.CONTAINS_ANY: "Include any of",
    ConstraintOp.CONTAINS_ALL: "Include all of",
    ConstraintOp.NOT_CONTAINS: "Must not include",
    ConstraintOp.NOT_CONTAINS_ANY: "Exclude any of",
    ConstraintOp.NOT_CONTAINS_ALL: "Not all of",
    ConstraintOp.SUBSTRING: "Text contains",
    ConstraintOp.BOOLEAN: "True/False",
    ConstraintOp.RANGE: "Between",
}


def _constraint_to_param(spec: ConstraintSpec) -> tuple[str, dict[str, Any]]:
    """Convert a ConstraintSpec into a JSON Schema property for the filter tool."""
    json_type = _VALUE_TYPE_TO_JSON.get(spec.value_type, "string")

    prop: dict[str, Any] = {"type": json_type}

    hint = _OP_HINT.get(spec.operator, "")
    desc = spec.display_template.replace("{value}", "…")
    prop["description"] = f"{hint}: {desc}" if hint else desc

    if spec.operator == ConstraintOp.IN_SET and spec.sampling_values:
        prop["enum"] = spec.sampling_values
    if spec.operator == ConstraintOp.EQ and spec.sampling_values and len(spec.sampling_values) <= 20:
        prop["enum"] = spec.sampling_values

    return spec.name, prop


# -- Fuzzy value matching ------------------------------------------------------

def _build_value_index(config: DomainConfig) -> dict[str, list[str]]:
    """Build a mapping from constraint name to all known valid values (lowercased)."""
    index: dict[str, list[str]] = {}
    for cspec in config.get_filterable_constraints():
        if cspec.sampling_values:
            index[cspec.name] = [str(v).lower() for v in cspec.sampling_values]
    return index


def _strip_prefix(val: str) -> str:
    """Remove common tag prefixes like 'en:', 'xx:', 'fr:'."""
    m = re.match(r"^[a-z]{2}:", val)
    return val[3:] if m else val


def _fuzzy_match_value(
    user_value: str,
    valid_values: list[str],
    original_values: list[Any],
) -> Any | None:
    """Find the best fuzzy match for a user-supplied value against known valid values.

    Matching cascade:
      1. Exact (case-insensitive)
      2. Prefix-stripped exact (e.g. "snacks" matches "en:snacks")
      3. Substring containment (user_value in valid or valid in user_value)
      4. SequenceMatcher ratio >= 0.7
    Returns the original (cased) value or None.
    """
    user_lower = user_value.lower().strip()
    if not user_lower:
        return None

    for i, vl in enumerate(valid_values):
        if user_lower == vl:
            return original_values[i]

    user_stripped = _strip_prefix(user_lower)
    for i, vl in enumerate(valid_values):
        vl_stripped = _strip_prefix(vl)
        if user_stripped == vl_stripped:
            return original_values[i]

    for i, vl in enumerate(valid_values):
        vl_stripped = _strip_prefix(vl)
        if user_stripped in vl_stripped or vl_stripped in user_stripped:
            return original_values[i]

    best_ratio = 0.0
    best_idx = -1
    for i, vl in enumerate(valid_values):
        vl_stripped = _strip_prefix(vl)
        ratio = SequenceMatcher(None, user_stripped, vl_stripped).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_ratio >= 0.7 and best_idx >= 0:
        return original_values[best_idx]

    return None


def _fuzzy_resolve_constraints(
    raw_constraints: dict[str, Any],
    config: DomainConfig,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve user-supplied constraint values via fuzzy matching.

    Returns (resolved_constraints, correction_notes) where correction_notes
    lists human-readable strings about values that were auto-corrected.
    """
    resolved: dict[str, Any] = {}
    notes: list[str] = []

    registry = {c.name: c for c in config.constraints}

    for key, val in raw_constraints.items():
        cspec = registry.get(key)
        if cspec is None or not cspec.sampling_values:
            resolved[key] = val
            continue

        if cspec.value_type in ("float", "int"):
            resolved[key] = val
            continue

        valid_lower = [str(v).lower() for v in cspec.sampling_values]
        originals = list(cspec.sampling_values)

        if isinstance(val, str):
            if val.lower() in valid_lower:
                resolved[key] = originals[valid_lower.index(val.lower())]
            else:
                matched = _fuzzy_match_value(val, valid_lower, originals)
                if matched is not None:
                    notes.append(f"Interpreted '{val}' as '{matched}' for {key}")
                    resolved[key] = matched
                else:
                    resolved[key] = val
        elif isinstance(val, list):
            resolved_list = []
            for item in val:
                item_str = str(item)
                if item_str.lower() in valid_lower:
                    resolved_list.append(originals[valid_lower.index(item_str.lower())])
                else:
                    matched = _fuzzy_match_value(item_str, valid_lower, originals)
                    if matched is not None:
                        notes.append(f"Interpreted '{item_str}' as '{matched}' for {key}")
                        resolved_list.append(matched)
                    else:
                        resolved_list.append(item)
            resolved[key] = resolved_list
        else:
            resolved[key] = val

    return resolved, notes


# -- Filter tool ---------------------------------------------------------------

def build_domain_filter_tool(config: DomainConfig) -> DomainToolSpec:
    filterable = config.get_filterable_constraints()

    properties: dict[str, Any] = {}
    param_descs: dict[str, str] = {}
    for cspec in filterable:
        pname, prop = _constraint_to_param(cspec)
        properties[pname] = prop
        param_descs[pname] = prop.get("description", "")

    properties["top_k"] = {
        "type": "integer",
        "description": "Maximum results to return (default 30)",
    }

    schema = {
        "type": "function",
        "function": {
            "name": "filter_products",
            "description": (
                f"Filter the {config.item_noun} catalog by structured constraints. "
                f"Returns {config.item_noun_plural} matching ALL specified criteria. "
                "Only specify constraints the user has told you about. "
                "Values are fuzzy-matched: you can use natural language "
                "(e.g. 'snacks' will match 'en:snacks')."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [],
            },
        },
    }

    constraint_registry = config.constraints

    def _filter_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        top_k = min(int(kwargs.pop("top_k", 30)), 200)
        raw_constraints = {k: v for k, v in kwargs.items() if v is not None}
        if not raw_constraints:
            sample = ctx.catalog.head(top_k)
            return sample.to_json(orient="records", indent=2) if len(sample) else "[]"

        constraints, corrections = _fuzzy_resolve_constraints(raw_constraints, ctx.config)

        filtered = GenericFilter.apply(ctx.catalog, constraints, constraint_registry)

        if len(filtered) == 0 and corrections:
            preamble = "Note: " + "; ".join(corrections) + "\n"
            return preamble + "No products matched all constraints. Try relaxing one."

        if len(filtered) == 0:
            return "[]"

        display_cols = [c for c in filtered.columns if not c.startswith("_")]
        result = filtered[display_cols].head(top_k).to_json(orient="records", indent=2)
        if corrections:
            result = "Note: " + "; ".join(corrections) + "\n" + result
        return result

    return DomainToolSpec(
        name="filter_products",
        description=schema["function"]["description"],
        fn=_filter_fn,
        schema=schema,
        param_descriptions=param_descs,
    )


# -- Detail tool ---------------------------------------------------------------

def build_domain_detail_tool(config: DomainConfig) -> DomainToolSpec:
    id_col = config.id_column
    id_attr = next((a for a in config.attributes if a.name == id_col), None)
    display_name = id_attr.display_name if id_attr else id_col.replace("_", " ").title()

    param_name = id_col
    param_desc = (
        f"The exact {display_name} as shown in search/filter results"
    )
    tool_desc = (
        f"Get detailed information about a specific {config.item_noun} "
        f"by its exact {display_name}."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": tool_desc,
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": "string",
                        "description": param_desc,
                    }
                },
                "required": [param_name],
            },
        },
    }

    attr_lookup = {a.name: a for a in config.attributes}

    def _detail_fn(ctx: DomainCatalogContext, **kwargs: Any) -> str:
        product_id = kwargs.get(param_name, kwargs.get("product_id"))
        if not product_id:
            return f"Missing required {param_name}."
        product = ctx.get_product(product_id)
        if product is None:
            return f"{config.item_noun.title()} '{product_id}' not found in catalog."
        lines: list[str] = [f"{display_name}: {product_id}"]
        for col, val in product.items():
            if val is None:
                continue
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            attr = attr_lookup.get(str(col))
            label = attr.display_name if attr else str(col).replace("_", " ").title()
            unit = f" {attr.unit}" if attr and attr.unit else ""
            if isinstance(val, (list, tuple)):
                val_str = ", ".join(str(v) for v in val)
            else:
                val_str = str(val)
            lines.append(f"{label}: {val_str}{unit}")
        return "\n".join(lines)

    return DomainToolSpec(
        name="get_product_details",
        description=schema["function"]["description"],
        fn=_detail_fn,
        schema=schema,
    )


# -- Search tool (simple text matching) ----------------------------------------

def _tokenize_for_search(text: str) -> list[str]:
    """Split text into normalized tokens: lowercase, split on non-alnum, strip tag prefixes."""
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower())
    tokens = []
    for t in raw:
        if not t or len(t) < 2:
            continue
        tokens.append(t)
        stripped = _strip_prefix(t)
        if stripped != t:
            tokens.append(stripped)
    return tokens


def build_domain_search_tool(config: DomainConfig) -> DomainToolSpec:
    all_searchable = []
    for a in config.attributes:
        if a.embedding_field:
            all_searchable.append(a.name)
        elif a.attr_type in (AttrType.TEXT, AttrType.CATEGORICAL, AttrType.SET_VALUED):
            all_searchable.append(a.name)

    if not all_searchable:
        all_searchable = [a.name for a in config.attributes][:8]

    schema = {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                f"Search the {config.item_noun} catalog using a natural language query. "
                f"Returns {config.item_noun_plural} ranked by text relevance. "
                "Supports partial and fuzzy matching across all text, categorical, and tag fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"Natural language search query for {config.item_noun_plural}",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default 10, max 30)",
                    },
                },
                "required": ["query"],
            },
        },
    }

    def _search_fn(ctx: DomainCatalogContext, query: str, top_k: int = 10) -> str:
        top_k = min(max(1, int(top_k)), 30)
        available = [c for c in all_searchable if c in ctx.catalog.columns]
        if not available:
            return "[]"

        catalog = ctx.catalog
        n_docs = len(catalog)

        query_tokens = _tokenize_for_search(query)
        if not query_tokens:
            return "[]"

        doc_freq: Counter = Counter()
        row_token_cache: dict[Any, set[str]] = {}

        sample_size = min(n_docs, 5000)
        if sample_size < n_docs:
            sample_idx = catalog.sample(sample_size, random_state=42).index
        else:
            sample_idx = catalog.index

        for idx in sample_idx:
            row = catalog.loc[idx]
            row_tokens: set[str] = set()
            for c in available:
                v = row.get(c) if isinstance(row, pd.Series) else row[c] if c in row else None
                if v is None:
                    continue
                try:
                    if isinstance(v, float) and pd.isna(v):
                        continue
                except (TypeError, ValueError):
                    pass
                if isinstance(v, (list, tuple, np.ndarray)):
                    text = " ".join(str(x) for x in v)
                else:
                    text = str(v)
                row_tokens.update(_tokenize_for_search(text))
            row_token_cache[idx] = row_tokens
            for qt in query_tokens:
                if qt in row_tokens:
                    doc_freq[qt] += 1

        idf: dict[str, float] = {}
        for qt in query_tokens:
            df = doc_freq.get(qt, 0)
            idf[qt] = np.log((sample_size + 1) / (df + 1)) + 1.0

        scores: list[tuple[float, Any]] = []
        for idx in catalog.index:
            if idx in row_token_cache:
                row_tokens = row_token_cache[idx]
            else:
                row = catalog.loc[idx]
                row_tokens = set()
                for c in available:
                    v = row.get(c) if isinstance(row, pd.Series) else None
                    if v is None:
                        continue
                    try:
                        if isinstance(v, float) and pd.isna(v):
                            continue
                    except (TypeError, ValueError):
                        pass
                    if isinstance(v, (list, tuple, np.ndarray)):
                        text = " ".join(str(x) for x in v)
                    else:
                        text = str(v)
                    row_tokens.update(_tokenize_for_search(text))

            score = 0.0
            for qt in query_tokens:
                if qt in row_tokens:
                    score += idf.get(qt, 1.0)
                else:
                    for rt in row_tokens:
                        if qt in rt or rt in qt:
                            score += idf.get(qt, 1.0) * 0.5
                            break

            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: -x[0])
        top_ids = [s[1] for s in scores[:top_k]]
        if not top_ids:
            return "[]"
        display_cols = [c for c in catalog.columns if not c.startswith("_")]
        return catalog.loc[top_ids, display_cols].to_json(orient="records", indent=2)

    return DomainToolSpec(
        name="search_products",
        description=schema["function"]["description"],
        fn=_search_fn,
        schema=schema,
    )


# -- Terminal tools (reused from Query2Cart, adapted for DomainCatalogContext) -

def _build_recommend_tool(config: DomainConfig | None = None) -> DomainToolSpec:
    id_col = config.id_column if config else "product_id"
    item_noun = config.item_noun if config else "product"
    item_noun_plural = config.item_noun_plural if config else "products"

    id_attr = None
    if config:
        id_attr = next((a for a in config.attributes if a.name == id_col), None)
    display_name = id_attr.display_name if id_attr else id_col.replace("_", " ").title()

    param_name = id_col + "s" if not id_col.endswith("s") else id_col
    param_desc = (
        f"Ordered list of exact {display_name} values to recommend (best first). "
        f"Use the {display_name} exactly as they appear in search/filter results."
    )

    schema = {
        "type": "function",
        "function": {
            "name": "recommend_products",
            "description": (
                f"Submit your final {item_noun} recommendation(s) to the user. "
                "This ENDS the conversation and your picks will be scored. "
                "Only call this when you are confident."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": param_desc,
                    }
                },
                "required": [param_name],
            },
        },
    }
    return DomainToolSpec(
        name="recommend_products",
        description=schema["function"]["description"],
        fn=lambda ctx, **kw: "Recommendation submitted.",
        schema=schema,
    )


def _build_declare_infeasible_tool() -> DomainToolSpec:
    schema = {
        "type": "function",
        "function": {
            "name": "declare_infeasible",
            "description": (
                "Declare that the user's requirements cannot be satisfied "
                "by any product in the catalog. Call this when you have "
                "thoroughly explored the catalog and determined that no "
                "product meets ALL of the user's stated constraints. "
                "This ENDS the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why the requirements are infeasible",
                    }
                },
                "required": ["reason"],
            },
        },
    }
    return DomainToolSpec(
        name="declare_infeasible",
        description=schema["function"]["description"],
        fn=lambda ctx, **kw: "Infeasibility declaration submitted.",
        schema=schema,
    )


# -- List filter values tool ---------------------------------------------------

def build_domain_list_values_tool(config: DomainConfig) -> DomainToolSpec:
    """Tool that lets the agent discover valid filter values for any constraint."""
    filterable = config.get_filterable_constraints()
    constraint_names = [c.name for c in filterable if c.sampling_values]

    schema = {
        "type": "function",
        "function": {
            "name": "list_filter_values",
            "description": (
                f"List the valid filter values for a {config.item_noun} constraint. "
                "Use this to discover what values are available before filtering. "
                f"Available constraints: {', '.join(constraint_names)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraint_name": {
                        "type": "string",
                        "description": "The constraint to list values for",
                        "enum": constraint_names,
                    },
                },
                "required": ["constraint_name"],
            },
        },
    }

    constraint_registry = {c.name: c for c in filterable}

    def _list_fn(ctx: DomainCatalogContext, constraint_name: str) -> str:
        cspec = constraint_registry.get(constraint_name)
        if cspec is None:
            available = ", ".join(constraint_names)
            return f"Unknown constraint '{constraint_name}'. Available: {available}"

        vals = cspec.sampling_values or []
        hint = _OP_HINT.get(cspec.operator, "")
        label = cspec.display_template.replace("{value}", "…")

        lines = [
            f"Constraint: {constraint_name}",
            f"Description: {hint}: {label}" if hint else f"Description: {label}",
            f"Operator: {cspec.operator.value}",
            f"Values ({len(vals)}):",
        ]
        for v in vals:
            lines.append(f"  - {v}")
        return "\n".join(lines)

    return DomainToolSpec(
        name="list_filter_values",
        description=schema["function"]["description"],
        fn=_list_fn,
        schema=schema,
    )


# -- Load domain-specific tools ------------------------------------------------

def load_domain_specific_tools(domain_dir: Path | str) -> list[DomainToolSpec]:
    """Load domain-specific tools from ``tools.py`` under *domain_dir*.

    Falls back to the legacy ``domain_specific_tools.py`` for backward
    compatibility.

    Returns an empty list if neither file exists or the module has no
    ``build_domain_specific_tools`` callable.
    """
    base = Path(domain_dir)
    tools_path = base / "tools.py"
    if not tools_path.is_file():
        tools_path = base / "domain_specific_tools.py"
    if not tools_path.is_file():
        return []

    spec = importlib.util.spec_from_file_location(
        f"domain_tools_{base.stem}",
        str(tools_path),
    )
    if spec is None or spec.loader is None:
        return []

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        logger.warning("Failed to load domain-specific tools from %s: %s", tools_path, exc)
        return []

    builder = getattr(mod, "build_domain_specific_tools", None)
    if builder is None:
        return []
    try:
        tools = builder()
    except Exception as exc:
        logger.warning("build_domain_specific_tools() failed in %s: %s", tools_path, exc)
        return []
    logger.info("Loaded %d domain-specific tools from %s", len(tools), tools_path)
    return tools


# -- Public API ----------------------------------------------------------------

def _build_terminal_tools(config: DomainConfig | None = None) -> list[DomainToolSpec]:
    return [
        _build_recommend_tool(config),
        _build_declare_infeasible_tool(),
    ]


TERMINAL_TOOLS: list[DomainToolSpec] = _build_terminal_tools()


def build_domain_tools(
    config: DomainConfig,
    domain_dir: str | Path | None = None,
) -> list[DomainToolSpec]:
    """Build the complete tool set for a domain.

    When domain-specific tools exist in ``domain_dir/tools.py``
    (or legacy ``domain_dir/domain_specific_tools.py``), uses ONLY those
    plus the terminal tools (recommend, declare_infeasible).

    Falls back to generic tools (filter, search, list_values, detail) when no
    domain-specific tools are available.
    """
    specific: list[DomainToolSpec] = []
    if domain_dir is not None:
        specific = load_domain_specific_tools(domain_dir)

    if specific:
        tools = list(specific)
    else:
        tools = [
            build_domain_filter_tool(config),
            build_domain_search_tool(config),
            build_domain_list_values_tool(config),
            build_domain_detail_tool(config),
        ]

    tools.extend(_build_terminal_tools(config))
    return tools


def get_domain_tool_schemas(tools: list[DomainToolSpec]) -> list[dict[str, Any]]:
    return [t.schema for t in tools]


def get_domain_tool_map(tools: list[DomainToolSpec]) -> dict[str, DomainToolSpec]:
    return {t.name: t for t in tools}
