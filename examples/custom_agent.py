"""Example: plug your own agent into the benchmark.

Usage:
    python examples/custom_agent.py
"""

from typing import Any

from simulation.benchmark import Benchmark
from simulation.agents.base import ConversationalAgent
from shared.types import AgentMessage, ToolCallInfo
from shared.llm import completion


class MyAgent(ConversationalAgent):
    """Minimal custom agent --- replace with your own logic."""

    def __init__(self, model: str = "gpt-4.1"):
        super().__init__()
        self.model = model
        self.tool_schemas = None

    def reset(self, env=None) -> None:
        self._messages = [
            {"role": "system", "content": "You are a helpful shopping assistant. Be concise."}
        ]
        if env is not None:
            self.tool_schemas = env.get_tool_schemas()

    def get_system_prompt(self) -> str:
        return self._messages[0]["content"]

    def act(self, message: dict[str, Any] | None = None) -> AgentMessage:
        if message is not None:
            self._messages.append(message)

        response = completion(
            model=self.model,
            messages=self._messages,
            tools=self.tool_schemas,
            temperature=0.0,
        )

        choice = response.choices[0].message
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if choice.content:
            assistant_msg["content"] = choice.content
        if choice.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.tool_calls
            ]
        self._messages.append(assistant_msg)

        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                ToolCallInfo(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in choice.tool_calls
            ]
        return AgentMessage(content=choice.content, tool_calls=tool_calls)


# Run on a pre-built domain
bench = Benchmark.load("games")
results = bench.run(agent_fn=lambda: MyAgent(model="gpt-4.1"))
bench.report(results)
