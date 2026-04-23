"""Prompt formatting for chat-style and plain-text model input.

Provides PromptFormatter implementations for converting structured
message lists into model-consumable prompt strings.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenizerProto(Protocol):
    """Minimum tokenizer surface required by ChatTemplateFormatter.

    Documents the duck-typed contract on HuggingFace tokenizers used
    by ``ChatTemplateFormatter``. The real ``transformers.PreTrainedTokenizer``
    structurally satisfies this Protocol.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        /,
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        **kwargs: Any,
    ) -> str: ...


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

    def __init__(self, tokenizer: TokenizerProto) -> None:
        self._tokenizer = tokenizer

    def format_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs: object,
    ) -> str:
        """Format messages using the tokenizer's chat template.

        Extra kwargs (e.g. enable_thinking for Qwen3.5) are passed
        through to apply_chat_template. Unsupported kwargs are silently
        dropped to maintain compatibility across model families.
        """
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TypeError:
            # Model template doesn't support these kwargs — retry without
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
