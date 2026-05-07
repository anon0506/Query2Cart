"""ToolCallingAgent --- default LLM-based conversational agent.

The system prompt is derived from the DomainConfig so it works for any domain.
"""

from __future__ import annotations

import logging
from typing import Any

from simulation.agents.base import ConversationalAgent
from shared.types import AgentMessage, ToolCallInfo
from shared.llm import completion, completion_cost

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_TEMPLATE = """\
You are a shopping assistant helping a customer find the right {item_noun} through conversation.

TOOLS:
You have access to a {item_noun} catalog with structured fields you can filter on. \
Use your tools to search, filter, and inspect {item_noun_plural}. \
Always check product details before recommending.

STRATEGY:
1. Read the customer's opening message for any stated needs or preferences.
2. Ask clarifying questions AND show early candidates — do both together; prioritize \
learning from the customer over rushing to finalize.
3. Use search for exploration, filters for narrowing by known constraints.
4. Use product details to verify ALL stated requirements before presenting.
5. Present 1-3 options, ask for feedback, and refine.
6. Only after you have asked as many useful questions as the dialogue warrants, \
and search backs your picks, call recommend_products with your final picks.

RULES:
- Before calling recommend_products or declare_infeasible, keep asking the customer \
clarifying questions until you have reasonably exhausted what they can tell you about \
needs, constraints, and trade-offs. Do not use a terminal tool while obvious or \
high-value questions remain unasked or unanswered.
- Never recommend a product without verifying it meets all stated constraints.
- If requirements seem impossible, explain the trade-off and suggest relaxation.
- Only call declare_infeasible after thorough search and after similarly thorough \
questioning — not as a shortcut when you could still learn more from the customer.
- Be concise — keep messages to 2-4 sentences.
- Call recommend_products exactly once, when confident.
- If you see [SYSTEM: Turn limit reached...], immediately call recommend_products."""


class ToolCallingAgent(ConversationalAgent):
    """LLM agent that uses tool calling to converse and search the catalog."""

    def __init__(
        self,
        config=None,
        model: str = "gpt-4.1",
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.total_cost: float = 0.0
        self.tool_schemas: list[dict] | None = None

        if system_prompt is not None:
            self._system_prompt = system_prompt
        elif config is not None:
            self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                item_noun=config.item_noun,
                item_noun_plural=config.item_noun_plural,
            )
        else:
            self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                item_noun="product",
                item_noun_plural="products",
            )

    def reset(self, env=None) -> None:
        """Store tool schemas and initialize message history."""
        self._messages = [{"role": "system", "content": self._system_prompt}]
        if env is not None:
            self.tool_schemas = env.get_tool_schemas()

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def act(self, message: dict[str, Any] | None = None) -> AgentMessage:
        if message is not None:
            self._messages.append(message)

        response = completion(
            model=self.model,
            messages=self._messages,
            tools=self.tool_schemas,
            temperature=self.temperature,
        )

        choice = response.choices[0].message
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if choice.content:
            assistant_msg["content"] = choice.content
        if choice.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.tool_calls
            ]
        self._messages.append(assistant_msg)

        cost = 0.0
        if hasattr(response, "_hidden_params"):
            cost = response._hidden_params.get("response_cost", 0) or 0
        if not cost and hasattr(response, "usage") and response.usage:
            cost = completion_cost(completion_response=response)

        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                ToolCallInfo(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in choice.tool_calls
            ]
        return AgentMessage(content=choice.content, tool_calls=tool_calls, cost=cost)
