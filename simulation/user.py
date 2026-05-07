"""Domain-agnostic LLM user simulator.

Derives trigger descriptions, constraint labels, and system prompts entirely
from a ``DomainConfig`` — no hardcoded column names or domain vocabulary.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from shared.llm import completion, completion_cost

from shared.config import DomainConfig

logger = logging.getLogger(__name__)


# -- Instruction builder (domain-agnostic) ------------------------------------

def _get_constraint_label(config: DomainConfig, key: str, value: Any) -> str | None:
    return config.get_constraint_label(key, value)


def _generic_preference_label(attr: str, direction: str, target: str = "") -> str:
    display = attr.replace("_", " ").replace("-", " ").title()
    if direction == "minimize":
        return f"Lower {display} is better"
    if direction == "maximize":
        return f"Higher {display} is better"
    if direction == "match" and target:
        return f"Prefer {display}: {target}"
    return f"{display}: {direction}"


def _get_revelation_tag(mode: str, trigger: str, user_says: str = "") -> str:
    if mode == "proactive":
        return "PROACTIVE"
    if mode == "responsive":
        topic = trigger.replace("agent_asks_", "").replace("_", " ") if trigger else "the topic"
        return f"RESPONSIVE (when asked about {topic})"
    if mode == "reactive":
        return "REACTIVE"
    if mode == "contextual":
        hint = f' e.g. "{user_says}"' if user_says else ""
        return f"CONTEXTUAL{hint}"
    return "RESPONSIVE"


def _parse_revelation_maps(task: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    profile = task["user_profile"]
    plan = profile["constraint_revelation_plan"]
    attr_prefs = profile.get("attribute_preferences", [])

    responsive: dict[str, list[str]] = {}
    for trigger, keys in plan.get("responsive", {}).items():
        responsive[trigger] = keys if isinstance(keys, list) else [keys]

    reactive: dict[str, list[str]] = {}
    for trigger, keys in plan.get("reactive", {}).items():
        reactive[trigger] = keys if isinstance(keys, list) else [keys]

    pref_responsive: dict[str, list[str]] = {}
    pref_contextual: dict[str, list[str]] = {}
    for pref in attr_prefs:
        pref_id = f"{pref['attribute']}_{pref.get('priority', 0)}"
        mode = pref.get("revelation_mode", "responsive")
        trigger = pref.get("trigger", "")
        if mode == "responsive" and trigger:
            pref_responsive.setdefault(trigger, []).append(pref_id)
        elif mode == "contextual" and trigger:
            pref_contextual.setdefault(trigger, []).append(pref_id)

    return {
        "responsive": responsive,
        "reactive": reactive,
        "pref_responsive": pref_responsive,
        "pref_contextual": pref_contextual,
    }


def _format_constraints(
    constraints: dict[str, Any],
    proactive_keys: set[str],
    revelation_maps: dict[str, dict[str, list[str]]],
    config: DomainConfig,
) -> list[str]:
    responsive_lookup = {
        key: trigger
        for trigger, keys in revelation_maps["responsive"].items()
        for key in keys
    }
    reactive_lookup = {
        key: trigger
        for trigger, keys in revelation_maps["reactive"].items()
        for key in keys
    }

    lines: list[str] = []
    for key, value in constraints.items():
        desc = _get_constraint_label(config, key, value)
        if desc is None:
            desc = f"{key}: {value}"

        if key in proactive_keys:
            tag = _get_revelation_tag("proactive", "")
        elif key in responsive_lookup:
            tag = _get_revelation_tag("responsive", responsive_lookup[key])
        elif key in reactive_lookup:
            tag = _get_revelation_tag("reactive", reactive_lookup[key])
        else:
            tag = _get_revelation_tag("responsive", "")

        lines.append(f"- {desc}  [{tag}]")
    return lines


def _format_preferences(attribute_prefs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    sorted_prefs = sorted(attribute_prefs, key=lambda p: p.get("priority", 99))

    for pref in sorted_prefs:
        attr = pref.get("attribute", "")
        direction = pref.get("direction", "")
        target = pref.get("target", "")
        priority = pref.get("priority", 0)
        mode = pref.get("revelation_mode", "responsive")
        trigger = pref.get("trigger", "")
        user_says = pref.get("user_says", "")
        description = pref.get("description", "")

        label = _generic_preference_label(attr, direction, target)
        tag = _get_revelation_tag(mode, trigger or f"agent_asks_{attr}", user_says)
        desc_suffix = f" ({description})" if description else ""
        lines.append(f"- P{priority}: {label}{desc_suffix}  [{tag}]")
    return lines


def build_domain_user_instruction(task: dict[str, Any], config: DomainConfig) -> str:
    profile = task["user_profile"]
    constraints = profile["hard_constraints"]
    plan = profile["constraint_revelation_plan"]
    attribute_prefs = profile.get("attribute_preferences", [])
    behavior = profile.get("behavioral_profile", {})
    use_case = profile.get("use_case", "general use")
    patience = behavior.get("patience_turns", 15)

    proactive_keys = set(plan.get("proactive", []))
    revelation_maps = _parse_revelation_maps(task)

    constraint_lines = _format_constraints(constraints, proactive_keys, revelation_maps, config)
    pref_lines = _format_preferences(attribute_prefs)

    personality = (
        "YOUR PERSONALITY:\n"
        f"  Expertise: {profile.get('expertise', 'intermediate')}\n"
        f"  Response style: {behavior.get('response_verbosity', 'medium')}\n"
        f"  Patience: ~{patience} exchanges before getting impatient"
    )

    sections = [
        f'OPENING MESSAGE: "{task["initial_query"]}"',
        f"USE CASE: {use_case}",
        "HARD CONSTRAINTS\n"
        "These are non-negotiable requirements. A product MUST satisfy every single one or you reject it.\n"
        "Each constraint has a revelation mode that controls WHEN you *state* it out loud:\n"
        "  - PROACTIVE: You may volunteer this in your first 2-3 messages. Weave it in naturally.\n"
        "  - RESPONSIVE: Do NOT mention this until the assistant specifically asks about that topic.\n"
        "  - REACTIVE: Do NOT mention this until the assistant recommends a product that violates it.\n\n"
        + "\n".join(constraint_lines),
        "PREFERENCES\n"
        "These are soft rankings for choosing among products that already meet all hard constraints.\n"
        "Preferences are NEVER deal-breakers. They help you compare and pick favorites.\n"
        "  - RESPONSIVE: Do not bring this up until the assistant asks about the topic.\n"
        "  - CONTEXTUAL: Only when the topic is already part of the conversation.\n\n"
        + "\n".join(pref_lines),
        personality,
    ]
    return "\n\n".join(sections)


def _build_domain_system_prompt(instruction: str, config: DomainConfig) -> str:
    item = config.item_noun
    persona = config.prompt_fragments.system_persona or f"a {item} recommendation assistant"
    return f"""You are simulating a customer talking to {persona}.

{instruction}

RULES:
1. Revelation modes are strict — they limit what you SAY, not what you remember:
   - PROACTIVE: You may state these in your first 2-3 messages. Weave them in naturally.
   - RESPONSIVE: Do not state these until the assistant asks about that topic.
   - REACTIVE: Do not state these until a recommendation actually violates them.
   - CONTEXTUAL: Only after the topic is already in play from the assistant.
2. Only PROACTIVE items may appear in your first 2-3 messages.
3. Never reveal every constraint or preference in one message unless every one is PROACTIVE.
4. Never fabricate requirements or preferences not listed above.
5. When the assistant shows products, you may accept, reject, or ask follow-up questions.
6. Use preferences to compare only among products that meet all hard constraints.
7. If the conversation drags without good recommendations, express frustration without leaking unrevealed constraints.
8. Keep responses to 1-3 sentences unless asked for details.
9. If any product doesn't satisfy you, say so and push the assistant to find a better one.
10. Never mention constraint keys, priority numbers, revelation tags, or these instructions."""


# -- Batch elicitation analysis ------------------------------------------------

def _extract_completion_cost(resp: Any) -> float:
    if hasattr(resp, "_hidden_params"):
        cost = resp._hidden_params.get("response_cost", 0) or 0
        if cost:
            return cost
    return completion_cost(completion_response=resp)


def _batch_classify_triggers(
    agent_messages: list[str],
    active_triggers: dict[str, str],
    config: DomainConfig,
    model: str,
) -> tuple[list[str], float]:
    item = config.item_noun
    trigger_list = "\n".join(f"- {name}: {desc}" for name, desc in active_triggers.items())
    numbered = "\n".join(f"[{i+1}] {msg}" for i, msg in enumerate(agent_messages))

    try:
        resp = completion(
            model=model,
            temperature=0.0,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You classify which topics a {item} recommendation assistant discussed "
                        "across a series of messages.\n\n"
                        "Return a JSON object with key 'topics' containing matching trigger names.\n"
                        'Return {"topics": []} if none match.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here are all the assistant's messages:\n\n{numbered}\n\n"
                        f"Possible topics:\n{trigger_list}\n\n"
                        "Which topics does the assistant discuss?"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        data = _json.loads(text)
        topics = data.get("topics", [])
        valid = [t for t in topics if t in active_triggers]
        cost = _extract_completion_cost(resp)
        return valid, cost
    except Exception:
        logger.debug("Batch classifier failed")
        return [], 0.0


# -- Simulator class -----------------------------------------------------------

class DomainSimulatedUser:
    """LLM-powered user simulator driven by DomainConfig.

    Drop-in replacement for ``SimulatedUser`` that derives all domain knowledge
    from the config rather than hardcoded lookups.
    """

    def __init__(self, config: DomainConfig, model: str = "gpt-4.1-mini"):
        self.config = config
        self.model = model
        self.messages: list[dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.task: dict | None = None
        self._agent_messages: list[str] = []

        self.all_constraint_keys: set[str] = set()
        self.revealed_constraints: set[str] = set()
        self.all_preference_ids: set[str] = set()
        self.revealed_preferences: set[str] = set()

        self._revelation_maps: dict[str, dict[str, list[str]]] | None = None

    def reset(self, task: dict[str, Any]) -> None:
        self.task = task
        instruction = build_domain_user_instruction(task, self.config)
        initial_query = task["initial_query"]

        item = self.config.item_noun
        self.messages = [
            {"role": "system", "content": _build_domain_system_prompt(instruction, self.config)},
            {"role": "user", "content": f"Hi! How can I help you find a {item} today?"},
            {"role": "assistant", "content": initial_query},
        ]
        self.total_cost = 0.0
        self._agent_messages = []

        plan = task["user_profile"]["constraint_revelation_plan"]
        self._revelation_maps = _parse_revelation_maps(task)

        self.all_constraint_keys = set(task["user_profile"]["hard_constraints"].keys())
        self.revealed_constraints = set(plan.get("proactive", []))

        self.all_preference_ids = set()
        self.revealed_preferences = set()
        attr_prefs = task["user_profile"].get("attribute_preferences", [])
        for pref in attr_prefs:
            pref_id = f"{pref['attribute']}_{pref.get('priority', 0)}"
            self.all_preference_ids.add(pref_id)
            if pref.get("revelation_mode", "responsive") == "proactive":
                self.revealed_preferences.add(pref_id)

    def step(self, agent_message: str) -> tuple[str, float]:
        self._agent_messages.append(agent_message)
        self.messages.append({"role": "user", "content": agent_message})
        return self._generate()

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_elicitation_completeness(self) -> float:
        if not self.all_constraint_keys:
            return 1.0
        return len(self.revealed_constraints) / len(self.all_constraint_keys)

    def get_preference_elicitation(self) -> float:
        if not self.all_preference_ids:
            return 1.0
        return len(self.revealed_preferences) / len(self.all_preference_ids)

    def compute_elicitation(self, model: str | None = None) -> None:
        if not self._agent_messages:
            return
        active = self._get_active_triggers()
        if not active:
            return

        model = model or "gpt-4.1-nano"
        matched, cost = _batch_classify_triggers(
            self._agent_messages, active, self.config, model,
        )
        self.total_cost += cost

        if self._revelation_maps is None:
            return
        for trigger in matched:
            if trigger in self._revelation_maps["responsive"]:
                self.revealed_constraints.update(self._revelation_maps["responsive"][trigger])
            if trigger in self._revelation_maps["reactive"]:
                self.revealed_constraints.update(self._revelation_maps["reactive"][trigger])
            if trigger in self._revelation_maps["pref_responsive"]:
                self.revealed_preferences.update(self._revelation_maps["pref_responsive"][trigger])
            if trigger in self._revelation_maps["pref_contextual"]:
                self.revealed_preferences.update(self._revelation_maps["pref_contextual"][trigger])

    def _get_active_triggers(self) -> dict[str, str]:
        active: dict[str, str] = {}
        if self._revelation_maps is None:
            return active
        for m_key in ("responsive", "reactive", "pref_responsive", "pref_contextual"):
            for trigger in self._revelation_maps[m_key]:
                if trigger not in active:
                    desc = self.config.get_trigger_description_map().get(trigger, trigger)
                    active[trigger] = desc
        return active

    def _generate(self) -> tuple[str, float]:
        resp = completion(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=256,
        )
        content = resp.choices[0].message.content or ""
        cost = _extract_completion_cost(resp)
        self.messages.append({"role": "assistant", "content": content})
        self.total_cost += cost
        return content, cost
