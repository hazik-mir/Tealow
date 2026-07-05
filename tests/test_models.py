"""Tests for TeaLow.models."""

from __future__ import annotations

from TeaLow.models import Conversation, Message, Response, StreamChunk, Usage


def test_message_to_dict_without_attachments() -> None:
    message = Message(role="user", content="hi")
    assert message.to_dict() == {"role": "user", "content": "hi"}


def test_message_to_dict_with_images_and_files() -> None:
    message = Message(
        role="user",
        content="describe this",
        images=[{"url": "http://example.com/a.png"}],
        files=[{"name": "a.txt", "base64": "abc"}],
        name="alice",
    )
    data = message.to_dict()
    assert data["images"] == [{"url": "http://example.com/a.png"}]
    assert data["files"] == [{"name": "a.txt", "base64": "abc"}]
    assert data["name"] == "alice"


def test_usage_from_dict_defaults_to_zero() -> None:
    usage = Usage.from_dict(None)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_usage_from_dict_parses_values() -> None:
    usage = Usage.from_dict(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_response_str_returns_text() -> None:
    response = Response(text="hello world", model="gpt-4o", provider="openai")
    assert str(response) == "hello world"
    assert f"{response}" == "hello world"


def test_response_to_dict() -> None:
    response = Response(
        text="hi",
        model="gpt-4o",
        provider="openai",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        finish_reason="stop",
    )
    data = response.to_dict()
    assert data["text"] == "hi"
    assert data["usage"]["total_tokens"] == 3
    assert data["finish_reason"] == "stop"


def test_stream_chunk_str_returns_delta() -> None:
    chunk = StreamChunk(delta="wor", model="gpt-4o", provider="openai")
    assert str(chunk) == "wor"


def test_conversation_add_and_history() -> None:
    convo = Conversation(system_prompt="be nice")
    convo.add_user_message("hi")
    convo.add_assistant_message("hello!")
    assert len(convo) == 2
    history = convo.as_list()
    assert history[0] == {"role": "system", "content": "be nice"}
    assert history[1]["role"] == "user"
    assert history[2]["role"] == "assistant"


def test_conversation_clear_keeps_system_prompt() -> None:
    convo = Conversation(system_prompt="be nice")
    convo.add_user_message("hi")
    convo.clear()
    assert len(convo) == 0
    assert convo.system_prompt == "be nice"


def test_conversation_as_list_without_system() -> None:
    convo = Conversation(system_prompt="be nice")
    convo.add_user_message("hi")
    history = convo.as_list(include_system=False)
    assert history[0]["role"] == "user"
