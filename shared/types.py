"""Core types for the Query2Cart environment."""

from dataclasses import dataclass, field
from typing import Any, Literal


RESPOND_ACTION_NAME = "respond_to_user"
RECOMMEND_ACTION_NAME = "recommend_products"
DECLARE_INFEASIBLE_ACTION_NAME = "declare_infeasible"


@dataclass
class Action:
    """An action taken by the agent: tool call, user message, or recommendation."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvResponse:
    """Response from the environment after an agent action.

    Follows the Gymnasium convention:
        observation, reward, done, truncated, info
    """
    observation: str
    reward: float
    done: bool              # done: natural episode end (recommend/infeasible)
    truncated: bool = False  # truncated: episode cut short (max turns hit)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallInfo:
    """A single tool call from the agent."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class SystemMessage:
    """System message for conversation context."""
    content: str
    role: Literal["system"] = "system"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class UserMessage:
    """User message in conversation."""

    content: str
    role: Literal["user"] = "user"
    # Bench metadata (not sent to the model API)
    llm_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class AssistantMessage:
    """Assistant/agent message with optional tool calls."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCallInfo] | None = None
    # Bench metadata (not sent to the model API)
    llm_cost: float = 0.0
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        msg_dict = {"role": self.role}
        if self.content is not None:
            msg_dict["content"] = self.content
        if self.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg_dict


@dataclass
class ToolMessage:
    """Tool response message."""

    tool_call_id: str
    content: str
    role: Literal["tool"] = "tool"
    # Bench metadata (not sent to the model API)
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self.content
        }


# Union type for all conversation messages
ConversationMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


@dataclass
class AgentMessage:
    """Response from an agent's act() call."""
    content: str | None = None
    tool_calls: list[ToolCallInfo] | None = None
    cost: float = 0.0


@dataclass
class SolveResult:
    """Result of an agent solving a single task."""
    task_id: str
    reward: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    recommended_products: list[str] = field(default_factory=list)
    conversation_turns: int = 0
    total_cost: float = 0.0
