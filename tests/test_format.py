"""Tests for tgirl.format — PromptFormatter implementations."""

from __future__ import annotations

import pytest


class TestPromptFormatterProtocol:
    """PromptFormatter is a runtime-checkable Protocol in types.py."""

    def test_protocol_is_runtime_checkable(self) -> None:
        from tgirl.types import PromptFormatter

        class MyFormatter:
            def format_messages(self, messages: list[dict[str, str]]) -> str:
                return ""

        assert isinstance(MyFormatter(), PromptFormatter)

    def test_non_conforming_class_rejected(self) -> None:
        from tgirl.types import PromptFormatter

        class NotAFormatter:
            pass

        assert not isinstance(NotAFormatter(), PromptFormatter)


class TestPlainFormatter:
    """PlainFormatter concatenates messages for base models / testing."""

    def test_empty_messages_returns_empty(self) -> None:
        from tgirl.format import PlainFormatter

        fmt = PlainFormatter()
        assert fmt.format_messages([]) == ""

    def test_single_message(self) -> None:
        from tgirl.format import PlainFormatter

        fmt = PlainFormatter()
        result = fmt.format_messages([{"role": "user", "content": "hello"}])
        assert "hello" in result

    def test_multiple_messages_concatenated(self) -> None:
        from tgirl.format import PlainFormatter

        fmt = PlainFormatter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = fmt.format_messages(messages)
        assert "You are helpful." in result
        assert "Hi there." in result
        assert "Hello!" in result

    def test_conforms_to_protocol(self) -> None:
        from tgirl.format import PlainFormatter
        from tgirl.types import PromptFormatter

        fmt = PlainFormatter()
        assert isinstance(fmt, PromptFormatter)


class TestChatTemplateFormatter:
    """ChatTemplateFormatter wraps HuggingFace tokenizer.apply_chat_template."""

    def test_delegates_to_tokenizer_apply_chat_template(self) -> None:
        from unittest.mock import MagicMock

        from tgirl.format import ChatTemplateFormatter

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<s>[INST] hello [/INST]"

        fmt = ChatTemplateFormatter(tokenizer)
        messages = [{"role": "user", "content": "hello"}]
        result = fmt.format_messages(messages)

        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert result == "<s>[INST] hello [/INST]"

    def test_add_generation_prompt_default_true(self) -> None:
        from unittest.mock import MagicMock

        from tgirl.format import ChatTemplateFormatter

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"

        fmt = ChatTemplateFormatter(tokenizer)
        fmt.format_messages([{"role": "user", "content": "hi"}])

        _, kwargs = tokenizer.apply_chat_template.call_args
        assert kwargs["add_generation_prompt"] is True

    def test_add_generation_prompt_false(self) -> None:
        from unittest.mock import MagicMock

        from tgirl.format import ChatTemplateFormatter

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"

        fmt = ChatTemplateFormatter(tokenizer)
        fmt.format_messages(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
        )

        _, kwargs = tokenizer.apply_chat_template.call_args
        assert kwargs["add_generation_prompt"] is False

    def test_conforms_to_protocol(self) -> None:
        from unittest.mock import MagicMock

        from tgirl.format import ChatTemplateFormatter
        from tgirl.types import PromptFormatter

        tokenizer = MagicMock()
        fmt = ChatTemplateFormatter(tokenizer)
        assert isinstance(fmt, PromptFormatter)
