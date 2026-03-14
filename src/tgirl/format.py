"""Prompt formatting for chat-style and plain-text model input.

Provides PromptFormatter implementations for converting structured
message lists into model-consumable prompt strings.
"""

from __future__ import annotations


class PlainFormatter:
    """Simple concatenation formatter for base models and testing.

    Joins messages as "role: content" lines separated by newlines.
    """

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return ""
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(parts)


class ChatTemplateFormatter:
    """Wraps a HuggingFace tokenizer's apply_chat_template method."""

    def __init__(self, tokenizer: object, **defaults: object) -> None:
        self._tokenizer = tokenizer
        self._defaults = defaults

    def format_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs: object,
    ) -> str:
        merged = {**self._defaults, **kwargs}
        return self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **merged,
        )
