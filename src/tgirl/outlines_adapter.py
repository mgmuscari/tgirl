"""Outlines/llguidance adapter for tgirl's GrammarState protocol.

Bridges llguidance's LLMatcher (used by Outlines for CFG constraint)
to tgirl's GrammarState protocol, enabling real grammar-constrained
inference via the tgirl sampling loop.

Optional dependency: requires ``outlines``, ``llguidance``, and
``transformers`` to be installed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import structlog
import torch

from tgirl.sample import GrammarState

logger = structlog.get_logger()

try:
    import llguidance
    import llguidance.hf
    import llguidance.torch as llg_torch
    from llguidance import LLMatcher
except ImportError as e:
    raise ImportError(
        "outlines_adapter requires llguidance. "
        "Install with: pip install 'tgirl[sample]' llguidance"
    ) from e


class LLGuidanceGrammarState:
    """GrammarState implementation backed by llguidance's LLMatcher.

    Satisfies the ``GrammarState`` protocol from ``tgirl.sample``:
    - ``get_valid_mask(vocab_size)`` -> boolean tensor
    - ``is_accepting()`` -> bool
    - ``advance(token_id)`` -> None
    """

    def __init__(
        self, matcher: LLMatcher, llg_vocab_size: int
    ) -> None:
        self._matcher = matcher
        self._llg_vocab_size = llg_vocab_size
        self._bitmask = llg_torch.allocate_token_bitmask(
            1, llg_vocab_size
        )

    def get_valid_mask(self, tokenizer_vocab_size: int) -> torch.Tensor:
        """Return a boolean mask of valid next tokens.

        The llguidance vocab size may differ from the model's vocab size
        (embedding padding, added tokens, etc.). This method handles the
        size mismatch by padding with False or truncating as needed.
        """
        llg_torch.fill_next_token_bitmask(
            self._matcher, self._bitmask, 0
        )
        logits = torch.zeros(self._llg_vocab_size)
        llg_torch.apply_token_bitmask_inplace(logits, self._bitmask[0])
        mask = logits > float("-inf")
        if tokenizer_vocab_size > self._llg_vocab_size:
            padding = torch.zeros(
                tokenizer_vocab_size - self._llg_vocab_size,
                dtype=torch.bool,
            )
            mask = torch.cat([mask, padding])
        elif tokenizer_vocab_size < self._llg_vocab_size:
            mask = mask[:tokenizer_vocab_size]
        return mask

    def get_valid_mask_np(self, tokenizer_vocab_size: int) -> np.ndarray:
        """Return a boolean mask of valid next tokens as numpy array.

        More efficient than get_valid_mask().numpy() when the consumer
        needs numpy (e.g., for conversion to mx.array in the MLX path).
        """
        mask_torch = self.get_valid_mask(tokenizer_vocab_size)
        return mask_torch.numpy()

    def is_accepting(self) -> bool:
        """Check if the grammar is in an accepting/final state."""
        return bool(self._matcher.is_accepting())

    def advance(self, token_id: int) -> None:
        """Consume a token and advance the grammar state."""
        self._matcher.consume_token(token_id)
        error = self._matcher.get_error()
        if error:
            logger.warning("llguidance_error", error=error, token_id=token_id)


def make_outlines_grammar_factory(
    tokenizer: object,
) -> Callable[[str], GrammarState]:
    """Create a grammar factory that produces GrammarState from Lark EBNF.

    Parameters
    ----------
    tokenizer
        A HuggingFace tokenizer (anything accepted by
        ``llguidance.hf.from_tokenizer``).

    Returns
    -------
    Callable[[str], GrammarState]
        Factory function: grammar_text -> GrammarState
    """
    llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
    llg_vocab_size: int = llg_tokenizer.vocab_size

    def factory(grammar_text: str) -> GrammarState:
        spec = llguidance.grammar_from("lark", grammar_text)
        matcher = LLMatcher(llg_tokenizer, spec)
        return LLGuidanceGrammarState(matcher, llg_vocab_size)

    return factory
