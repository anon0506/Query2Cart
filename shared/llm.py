"""Centralized LLM & embedding interface (litellm backend).

Uses **litellm** so any provider (OpenAI, Anthropic, Gemini, local models,
etc.) works out of the box --- just set the relevant API key env var.

This is the default backend for open-source users. 

Public API
----------
Text generation:
    call_llm(prompt, model, **kwargs) -> str
    call_llm_json(prompt, model, **kwargs) -> dict

Full chat completion (tool-calling, multi-turn):
    completion(**kwargs) -> response   (litellm response object)

Embeddings:
    call_embedding(input, model) -> list[list[float]]

Cost tracking:
    completion_cost(completion_response) -> float

Utilities:
    format_prompt_template, resolve_prompt_path, parse_json_response
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

for _ll in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
    logging.getLogger(_ll).setLevel(logging.WARNING)

_LLM_LOG_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

_GENERATION_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "generation" / "prompts"


# ---------------------------------------------------------------------------
# JSONL call logging (thread-safe)
# ---------------------------------------------------------------------------

def _append_llm_jsonl(record: dict[str, Any]) -> None:
    if (os.environ.get("Q2C_LLM_LOG_DISABLE") or "").lower() in (
        "1", "true", "yes", "on",
    ):
        return
    path = os.environ.get("Q2C_LLM_LOG_JSONL", "llm_calls.jsonl")
    if not path.strip():
        return
    line = json.dumps(record, ensure_ascii=False) + "\n"
    p = Path(path).expanduser()
    with _LLM_LOG_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Public API — completion / embeddings / cost
# ---------------------------------------------------------------------------

def completion(**kwargs: Any) -> Any:
    """Chat completion via litellm (drop-in passthrough).

    Accepts all ``litellm.completion()`` keyword arguments:
    model, messages, tools, temperature, max_tokens, response_format, etc.
    """
    return litellm.completion(**kwargs)


def completion_cost(completion_response: Any = None, **kwargs: Any) -> float:
    """Estimate cost of a completion via litellm."""
    try:
        return litellm.completion_cost(
            completion_response=completion_response, **kwargs,
        ) or 0.0
    except Exception:
        return 0.0


def call_embedding(
    input: str | list[str],
    model: str = "text-embedding-3-large",
) -> list[list[float]]:
    """Get embeddings via litellm."""
    if isinstance(input, str):
        input = [input]
    response = litellm.embedding(model=model, input=input)
    return [d["embedding"] for d in response.data]


# ---------------------------------------------------------------------------
# Public API — text generation helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str = "o4-mini", **kwargs: Any) -> str:
    """Call an LLM with a single user-message prompt via litellm."""
    temperature = float(kwargs.get("temperature", 0.7))
    max_tokens = int(kwargs.get("max_tokens", 32768))
    response_format = kwargs.get("response_format")

    call_id = str(uuid.uuid4())
    result_text: str | None = None
    error_text: str | None = None

    try:
        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        response = litellm.completion(**completion_kwargs)
        result_text = response.choices[0].message.content or ""
        return result_text
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        raise
    finally:
        record: dict[str, Any] = {
            "id": call_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "backend": "litellm",
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "result": result_text,
            "error": error_text,
        }
        if response_format is not None:
            record["response_format"] = response_format
        _append_llm_jsonl(record)


def call_llm_json(
    prompt: str,
    model: str,
    *,
    parse_json_retries: int = 3,
    **llm_kwargs: Any,
) -> dict[str, Any]:
    """Call :func:`call_llm` and parse the result as JSON with retries."""
    max_rounds = 1 + max(0, parse_json_retries)
    for attempt in range(1, max_rounds + 1):
        raw = call_llm(prompt, model, **llm_kwargs)
        try:
            return parse_json_response(raw)
        except json.JSONDecodeError as e:
            if attempt < max_rounds:
                print(
                    f"[llm] JSON parse failed (attempt {attempt}/{max_rounds}): {e!s}; "
                    "retrying LLM call...",
                )
                continue
            raise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def format_prompt_template(template: str, **kwargs: Any) -> str:
    """Fill ``{name}`` placeholders; unknown keys stay as literal ``{name}``."""

    class SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeDict(**{k: str(v) for k, v in kwargs.items()}))


def resolve_prompt_path(name: str) -> str:
    """Resolve a prompt template name to its absolute file path."""
    return str(_GENERATION_PROMPTS_DIR / name)


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response that may contain markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    return json.loads(text)
