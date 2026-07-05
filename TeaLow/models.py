"""Core domain models used by the TeaLow SDK.

These dataclasses represent conversation messages, normalized provider
responses, and streaming chunks. They are intentionally provider
agnostic: each concrete provider implementation is responsible for
translating its own wire format into these shared structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import FilePayload, ImagePayload, UsageDict


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The role of the message author. One of ``"system"``,
            ``"user"``, or ``"assistant"``.
        content: The textual content of the message.
        images: Optional list of image attachments for vision-capable
            models.
        files: Optional list of non-image file attachments.
        name: Optional name identifying the participant (used by some
            providers to distinguish between multiple users/tools).
    """

    role: str
    content: str
    images: List[ImagePayload] = field(default_factory=list)
    files: List[FilePayload] = field(default_factory=list)
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a plain dictionary representation."""
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            data["images"] = list(self.images)
        if self.files:
            data["files"] = list(self.files)
        if self.name:
            data["name"] = self.name
        return data


@dataclass
class Usage:
    """Token usage statistics for a single request/response cycle."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Optional[UsageDict]) -> "Usage":
        """Build a :class:`Usage` instance from a raw provider payload."""
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
        )


@dataclass
class Response:
    """Normalized response returned to the caller of :meth:`TeaLow.send`.

    The :attr:`text` attribute always contains the primary textual
    output. ``__str__`` returns this text directly so that
    ``print(response)`` behaves intuitively, as demonstrated in the
    project README.
    """

    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: Optional[str] = None
    raw: Any = None
    image_urls: List[str] = field(default_factory=list)
    image_base64: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response into a JSON-serializable dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "finish_reason": self.finish_reason,
            "image_urls": self.image_urls,
            "image_base64": self.image_base64,
        }


@dataclass
class StreamChunk:
    """A single incremental chunk emitted while streaming a response."""

    delta: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    raw: Any = None
    is_final: bool = False

    def __str__(self) -> str:
        return self.delta


class Conversation:
    """Maintains ordered conversation history for a :class:`TeaLow` client.

    The conversation keeps an optional system prompt separate from the
    turn-by-turn history so that providers with different system-prompt
    conventions (e.g. Anthropic's top-level ``system`` parameter versus
    OpenAI's ``system`` role message) can each consume it appropriately.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.system_prompt: Optional[str] = system_prompt
        self._messages: List[Message] = []

    @property
    def messages(self) -> List[Message]:
        """Return the list of non-system messages recorded so far."""
        return self._messages

    def add_user_message(
        self,
        content: str,
        images: Optional[List[ImagePayload]] = None,
        files: Optional[List[FilePayload]] = None,
    ) -> Message:
        """Append a user message to the conversation history."""
        message = Message(
            role="user",
            content=content,
            images=images or [],
            files=files or [],
        )
        self._messages.append(message)
        return message

    def add_assistant_message(self, content: str) -> Message:
        """Append an assistant message to the conversation history."""
        message = Message(role="assistant", content=content)
        self._messages.append(message)
        return message

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """Set or clear the system prompt for this conversation."""
        self.system_prompt = prompt

    def clear(self) -> None:
        """Remove all turn-by-turn history while keeping the system prompt."""
        self._messages.clear()

    def as_list(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Return the full conversation as a list of plain dictionaries.

        Args:
            include_system: Whether to prepend a system-role message
                dictionary when :attr:`system_prompt` is set.
        """
        result: List[Dict[str, Any]] = []
        if include_system and self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(message.to_dict() for message in self._messages)
        return result

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)
