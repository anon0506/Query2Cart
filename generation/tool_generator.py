"""LLM-based generation of domain-specific catalog tools.

For each domain the LLM analyses the catalog schema, constraints, and
preference attributes, then designs and implements the complete tool set
the recommendation agent will use.  Only ``recommend_products`` and
``declare_infeasible`` are fixed; every other tool is generated to fit the
specific dataset.

Each tool is generated one at a time with full context so the LLM can
produce high-quality, working implementations.  A single verification pass
at the end checks all tools against the actual catalog.

Usage (notebook)::

    from generation.tool_generator import generate_domain_specific_tools

    tools_path = generate_domain_specific_tools(
        config, catalog, DOMAIN_DIR, model="gpt-4.1",
    )
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.config import DomainConfig
from shared.llm import (
    call_llm,
    call_llm_json,
    format_prompt_template,
    resolve_prompt_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _fmt_val(val: Any, max_len: int = 80) -> str:
    """Format a single catalog cell for display in an LLM prompt."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "null"
    if isinstance(val, np.ndarray):
        items = val.tolist()
        if len(items) > 6:
            return f"[{', '.join(repr(x) for x in items[:6])}, ...] ({len(items)} items)"
        return f"[{', '.join(repr(x) for x in items)}]"
    if isinstance(val, (list, tuple)):
        if len(val) > 6:
            return f"[{', '.join(repr(x) for x in val[:6])}, ...] ({len(val)} items)"
        return f"[{', '.join(repr(x) for x in val)}]"
    if isinstance(val, str) and len(val) > max_len:
        return repr(val[: max_len - 3] + "...")
    return repr(val)


def _column_type_hint(catalog: pd.DataFrame, col: str) -> str:
    """Return a human-readable type hint for a catalog column."""
    non_null = catalog[col].dropna()
    if len(non_null) == 0:
        return "all-null"
    sample = non_null.iloc[0]
    if isinstance(sample, (np.ndarray, list, tuple)):
        return "array of strings"
    if isinstance(sample, bool) or catalog[col].dtype == bool:
        return "bool"
    if pd.api.types.is_float_dtype(catalog[col]):
        return "float"
    if pd.api.types.is_integer_dtype(catalog[col]):
        return "int"
    return "str"


def _build_domain_context(
    config: DomainConfig,
    catalog: pd.DataFrame,
    *,
    max_sample_rows: int = 3,
) -> str:
    """Build the full domain context string injected into every prompt."""
    lines: list[str] = []

    desc = (config.prompt_fragments.domain_description or "").strip() or config.name
    lines.append(f"DOMAIN: {desc}")
    lines.append(f"ITEM: {config.item_noun} (plural: {config.item_noun_plural})")
    lines.append(f"ID COLUMN: {config.id_column} (dtype: {_column_type_hint(catalog, config.id_column) if config.id_column in catalog.columns else str(catalog.index.dtype)})")
    lines.append(f"CATALOG: {len(catalog)} rows × {len(catalog.columns)} columns")
    lines.append("")

    lines.append(f"{'Column':<35s} {'Type':<18s} {'Non-null%':>9s}  Notes / sample")
    lines.append("─" * 95)
    for col in catalog.columns:
        non_null_pct = catalog[col].notna().sum() / len(catalog) * 100
        type_hint = _column_type_hint(catalog, col)
        non_null = catalog[col].dropna()
        if len(non_null) > 0:
            samples = non_null.sample(min(2, len(non_null)), random_state=42).tolist()
            sample_text = ", ".join(_fmt_val(s, 50) for s in samples)
        else:
            sample_text = "(all null)"
        lines.append(f"  {col:<33s} {type_hint:<18s} {non_null_pct:>8.1f}%  {sample_text}")
    lines.append("")

    filterable = config.get_filterable_constraints()
    lines.append(f"CONSTRAINTS ({len(filterable)} filterable):")
    for c in filterable:
        vals = ""
        if c.sampling_values:
            show = c.sampling_values[:5]
            vals = ", ".join(str(v) for v in show)
            if len(c.sampling_values) > 5:
                vals += f", ... ({len(c.sampling_values)} total)"
        lines.append(
            f"  {c.name:<38s} → {c.attribute:<28s} {c.operator.value:<14s} [{vals}]"
        )
    lines.append("")

    if config.preference_attributes:
        prefs = ", ".join(
            f"{p.attribute} ({', '.join(p.directions)})"
            for p in config.preference_attributes
        )
        lines.append(f"PREFERENCE ATTRIBUTES: {prefs}")
        lines.append("")

    sample = catalog.head(max_sample_rows)
    lines.append(f"SAMPLE ROWS ({len(sample)} rows):")
    for i, (idx, row) in enumerate(sample.iterrows()):
        lines.append(f"\n  Row {i} (index={_fmt_val(idx)}):")
        for col in catalog.columns:
            lines.append(f"    {col}: {_fmt_val(row[col])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompt(name: str) -> str:
    """Load a prompt template from ``domain/prompts/{name}``."""
    return Path(resolve_prompt_path(name)).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_code_block(response: str) -> str:
    """Extract Python code from an LLM response with code fences."""
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def _syntax_ok(code: str, label: str = "<generated>") -> bool:
    """Return True if *code* compiles without SyntaxError."""
    try:
        compile(code, label, "exec")
        return True
    except SyntaxError as exc:
        logger.warning("Syntax error in %s: %s", label, exc)
        return False


# ---------------------------------------------------------------------------
# Step 1 — Plan which tools to generate
# ---------------------------------------------------------------------------

def plan_domain_tools(
    config: DomainConfig,
    catalog: pd.DataFrame,
    *,
    model: str = "gpt-4.1",
    max_tools: int = 6,
) -> list[dict[str, Any]]:
    """Ask the LLM to analyse the domain and design the full tool set."""
    context = _build_domain_context(config, catalog)
    template = _load_prompt("plan_tools.txt")
    prompt = format_prompt_template(
        template,
        item_noun=config.item_noun,
        context=context,
        max_tools=max_tools,
    )
    data = call_llm_json(
        prompt, model, temperature=0.4, max_tokens=4096,
        response_format={"type": "json_object"},
    )
    tools = data.get("tools", [])
    logger.info("Tool planner proposed %d tools: %s",
                len(tools), [t.get("name") for t in tools])
    return tools


# ---------------------------------------------------------------------------
# Step 2 — Generate one tool at a time
# ---------------------------------------------------------------------------

def generate_single_tool(
    tool_plan: dict[str, Any],
    config: DomainConfig,
    catalog: pd.DataFrame,
    *,
    model: str = "gpt-4.1",
    existing_tool_names: list[str] | None = None,
    max_retries: int = 2,
) -> str:
    """Generate Python code for one tool.

    Returns the code string for a ``_build_<name>_tool()`` function.
    """
    context = _build_domain_context(config, catalog)
    id_col = config.id_column
    id_dtype = _column_type_hint(catalog, id_col) if id_col in catalog.columns else str(catalog.index.dtype)

    existing_note = ""
    if existing_tool_names:
        existing_note = (
            "Already-generated tools (avoid overlapping functionality):\n  "
            + ", ".join(existing_tool_names)
        )

    params_text = json.dumps(tool_plan.get("parameters", []), indent=2)
    template = _load_prompt("generate_tool.txt")
    prompt = format_prompt_template(
        template,
        item_noun=config.item_noun,
        context=context,
        name=tool_plan["name"],
        description=tool_plan["description"],
        parameters=params_text,
        returns=tool_plan.get("returns", "string"),
        rationale=tool_plan.get("rationale", ""),
        existing_note=existing_note,
        id_column=id_col,
        id_dtype=id_dtype,
    )

    for attempt in range(1, max_retries + 2):
        raw = call_llm(prompt, model, temperature=0.2, max_tokens=6000)
        code = _extract_code_block(raw)
        if _syntax_ok(code, tool_plan["name"]):
            return code
        if attempt <= max_retries:
            logger.warning(
                "Syntax error in tool '%s' (attempt %d/%d), retrying…",
                tool_plan["name"], attempt, max_retries + 1,
            )
    raise RuntimeError(
        f"Failed to generate syntactically valid code for tool '{tool_plan['name']}' "
        f"after {max_retries + 1} attempts"
    )


# ---------------------------------------------------------------------------
# Step 3 — Assemble module
# ---------------------------------------------------------------------------

_MODULE_HEADER = '''\
"""Auto-generated domain-specific tools for {item_noun_plural}.

Generated by domain.tool_generator — review before use.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from simulation.tools import DomainCatalogContext, DomainToolSpec


def build_domain_specific_tools() -> list[DomainToolSpec]:
    """Return all domain-specific tools for this catalog."""
    return [
{tool_list}
    ]


'''


def _assemble_module(
    tool_codes: list[str],
    tool_names: list[str],
    config: DomainConfig,
) -> str:
    """Combine individual tool codes into a complete Python module."""
    tool_list = "\n".join(f"        _build_{name}_tool()," for name in tool_names)
    header = _MODULE_HEADER.format(
        item_noun_plural=config.item_noun_plural,
        tool_list=tool_list,
    )
    body = "\n\n".join(tool_codes)
    module = header + body + "\n"
    return module


# ---------------------------------------------------------------------------
# Step 4 — Verification
# ---------------------------------------------------------------------------

def verify_domain_tools(
    module_code: str,
    config: DomainConfig,
    catalog: pd.DataFrame,
    *,
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    """Single LLM verification pass over all generated tools."""
    context = _build_domain_context(config, catalog, max_sample_rows=2)
    id_col = config.id_column
    id_dtype = _column_type_hint(catalog, id_col) if id_col in catalog.columns else str(catalog.index.dtype)

    template = _load_prompt("verify_tools.txt")
    prompt = format_prompt_template(
        template,
        item_noun=config.item_noun,
        context=context,
        module_code=module_code,
        id_dtype=id_dtype,
    )
    result = call_llm_json(
        prompt, model, temperature=0.1, max_tokens=16000,
        response_format={"type": "json_object"},
    )
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_domain_specific_tools(
    config: DomainConfig,
    catalog: pd.DataFrame,
    output_dir: Path | str,
    *,
    model: str = "gpt-4.1",
    max_tools: int = 6,
) -> Path | None:
    """Full pipeline: plan → generate each → assemble → verify → save.

    Returns the path to the saved ``domain_specific_tools.py``, or None on
    failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_path = output_dir / "domain_specific_tools.py"

    # ── Step 1: Plan ──────────────────────────────────────────────────────
    print(f"  [1/4] Planning tools for {config.item_noun} domain (model={model})…")
    plans = plan_domain_tools(config, catalog, model=model, max_tools=max_tools)
    if not plans:
        logger.warning("Tool planner returned no tools")
        return None

    print(f"         Proposed {len(plans)} tools:")
    for p in plans:
        print(f"           • {p['name']}: {p.get('description', '')[:80]}")

    # ── Step 2: Generate each tool ────────────────────────────────────────
    tool_codes: list[str] = []
    tool_names: list[str] = []
    for i, plan in enumerate(plans, 1):
        name = plan["name"]
        print(f"  [2/4] Generating tool {i}/{len(plans)}: {name}…")
        code = generate_single_tool(
            plan, config, catalog,
            model=model,
            existing_tool_names=tool_names,
        )
        tool_codes.append(code)
        tool_names.append(name)
        print(f"         ✓ {name} ({len(code)} chars)")

    # ── Step 3: Assemble ──────────────────────────────────────────────────
    print("  [3/4] Assembling module…")
    module_code = _assemble_module(tool_codes, tool_names, config)
    if not _syntax_ok(module_code, "domain_specific_tools.py"):
        logger.error("Assembled module has syntax errors — aborting")
        _save_debug(tools_path.with_suffix(".py.debug"), module_code)
        return None

    # ── Step 4: Verify ────────────────────────────────────────────────────
    print(f"  [4/4] Verifying all tools against catalog (model={model})…")
    verdict = verify_domain_tools(module_code, config, catalog, model=model)

    status = verdict.get("verdict", "unknown")
    reviews = verdict.get("tool_reviews", [])
    for r in reviews:
        icon = "✓" if r.get("status") == "pass" else "✗"
        issues = "; ".join(r.get("issues", []))
        severity = r.get("severity", "")
        detail = f" [{severity}] {issues}" if issues else ""
        print(f"         {icon} {r.get('name', '?')}{detail}")

    if status == "needs_fixes":
        fixed = verdict.get("fixed_module")
        if fixed and isinstance(fixed, str) and _syntax_ok(fixed, "fixed_module"):
            print("         Applied LLM-suggested fixes.")
            module_code = fixed
        else:
            print("         ⚠  Verification flagged issues but no valid fix provided. "
                  "Saving as-is — review the file manually.")

    tools_path.write_text(module_code, encoding="utf-8")
    print(f"\n  ✓ Saved domain-specific tools to: {tools_path}")
    return tools_path


def _save_debug(path: Path, code: str) -> None:
    """Save code to a debug file for inspection."""
    path.write_text(code, encoding="utf-8")
    logger.info("Debug output saved to %s", path)
