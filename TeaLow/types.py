"""Shared type definitions used across the TeaLow SDK.

This module defines lightweight ``TypedDict`` structures that describe
the shape of data exchanged with providers, plus a few type aliases
that make signatures elsewhere in the codebase easier to read.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:  # pragma: no cover - exercised on Python 3.9/3.10
    from typing_extensions import NotRequired, TypedDict


class ImagePayload(TypedDict, total=False):
    """Represents an image attached to a message, either inline or by URL."""

    url: NotRequired[str]
    base64: NotRequired[str]
    mime_type: NotRequired[str]


class FilePayload(TypedDict, total=False):
    """Represents a non-image file attached to a message."""

    path: NotRequired[str]
    base64: NotRequired[str]
    mime_type: NotRequired[str]
    name: NotRequired[str]


class MessageDict(TypedDict, total=False):
    """Dictionary representation of a single conversation message."""

    role: str
    content: str
    images: NotRequired[List[ImagePayload]]
    files: NotRequired[List[FilePayload]]
    name: NotRequired[str]


class UsageDict(TypedDict, total=False):
    """Token usage statistics reported by a provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseDict(TypedDict, total=False):
    """Normalized response payload returned by :meth:`TeaLow.send`."""

    text: str
    model: str
    provider: str
    usage: UsageDict
    raw: Any
    finish_reason: Optional[str]


JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
Headers = Dict[str, str]
