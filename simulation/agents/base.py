"""Abstract base agents for Query2Cart.

Two agent interfaces:
  - BaseAgent: non-conversational agents that implement act() to return
    one Action at a time. The orchestrator drives the step loop.
  - ConversationalAgent: LLM-based agents that implement act() to generate
    the next message; the orchestrator drives the conversation loop.
"""

from __future__ import annotations

import abc
from typing import Any

from shared.types import Action, AgentMessage


class BaseAgent(abc.ABC):
    """Base class for non-conversational agents.

    Subclasses implement act() to return the next Action given an observation.
    The orchestrator calls reset() then act() in a loop.
    """
    
    # Metadata for agent registry
    requires_llm: bool = False

    def reset(self, env=None) -> None:
        """Reset agent state for a new task.

        Called by the orchestrator before the step loop begins.
        Agents that need env context (e.g. oracle agents) can inspect
        env.current_task, env.catalog, etc.
        """
        pass

    def catalog_tool_names(self) -> tuple[str, ...]:
        """Catalog tools this agent may invoke (non-terminal); orchestrator builds ToolSpecs."""
        return ()

    @abc.abstractmethod
    def act(self, observation: str) -> Action:
        """Given an observation from the environment, return the next action."""
        ...


class ConversationalAgent(abc.ABC):
    """Base class for LLM-based conversational agents.

    Subclasses implement act() to produce the next AgentMessage.
    Agents now manage their own message history internally.
    """
    
    # Metadata for agent registry
    requires_llm: bool = True

    def __init__(self):
        # Internal message history for LLM context
        self._messages: list[dict[str, Any]] = []

    def reset(self, env=None) -> None:
        """Reset agent state for a new task.

        Called by the orchestrator before the conversation loop begins.
        Agents can inspect env.current_task, env.get_tool_schemas(), etc.
        """
        # Initialize with system prompt
        self._messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]

    def catalog_tool_names(self) -> tuple[str, ...]:
        """Catalog tools this agent may invoke (non-terminal); orchestrator builds ToolSpecs."""
        return ()

    def add_to_history(self, message: dict[str, Any]) -> None:
        """Add a message to the agent's internal history without generating a response."""
        self._messages.append(message)

    def get_messages(self) -> list[dict[str, Any]]:
        """Get the agent's current message history."""
        return list(self._messages)

    @abc.abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    @abc.abstractmethod
    def act(self, message: dict[str, Any] | None = None) -> AgentMessage:
        """Generate the next agent message, optionally processing a new incoming message.

        Args:
            message: New message to add to history (user response, tool result, etc.).
                    None for the first call after reset.

        The agent adds the message to its internal history, then generates a response.
        Tool schemas are provided via reset(env).
        """
        ...
