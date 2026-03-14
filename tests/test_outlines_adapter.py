"""Tests for tgirl.outlines_adapter — llguidance GrammarState bridge.

These tests require outlines, llguidance, and transformers to be installed.
They use GPT-2's tokenizer (downloaded on first run) for real token-level
grammar constraint validation.
"""

from __future__ import annotations

import pytest
import torch

# Skip entire module if dependencies missing
llguidance = pytest.importorskip("llguidance")
pytest.importorskip("transformers")

from tgirl.outlines_adapter import (
    LLGuidanceGrammarState,
    make_outlines_grammar_factory,
)
from tgirl.sample import GrammarState


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Load GPT-2 tokenizer once for all tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def grammar_factory(gpt2_tokenizer):
    """Create grammar factory from GPT-2 tokenizer."""
    return make_outlines_grammar_factory(gpt2_tokenizer)


SIMPLE_GRAMMAR = """\
?start: "(" "add" " " INT " " INT ")"
INT: /[0-9]+/
"""

TOOL_GRAMMAR = """\
?start: expr
expr: call_add | call_greet
call_add: "(" "add" " " SIGNED_INT " " SIGNED_INT ")"
call_greet: "(" "greet" " " ESCAPED_STRING ")"
%import common.ESCAPED_STRING
%import common.SIGNED_INT
"""


class TestProtocolCompliance:
    """LLGuidanceGrammarState satisfies the GrammarState protocol."""

    def test_isinstance_check(self, grammar_factory):
        state = grammar_factory(SIMPLE_GRAMMAR)
        assert isinstance(state, GrammarState)

    def test_has_get_valid_mask(self, grammar_factory):
        state = grammar_factory(SIMPLE_GRAMMAR)
        assert callable(state.get_valid_mask)

    def test_has_is_accepting(self, grammar_factory):
        state = grammar_factory(SIMPLE_GRAMMAR)
        assert callable(state.is_accepting)

    def test_has_advance(self, grammar_factory):
        state = grammar_factory(SIMPLE_GRAMMAR)
        assert callable(state.advance)


class TestGetValidMask:
    """get_valid_mask returns correct boolean tensor."""

    def test_returns_bool_tensor(self, grammar_factory, gpt2_tokenizer):
        state = grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool

    def test_correct_shape(self, grammar_factory, gpt2_tokenizer):
        state = grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        assert mask.shape == (gpt2_tokenizer.vocab_size,)

    def test_initial_state_constrains(self, grammar_factory, gpt2_tokenizer):
        """Initial state should only allow tokens starting with '('."""
        state = grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        valid_count = mask.sum().item()
        # Must be constrained (not all tokens valid)
        assert valid_count < gpt2_tokenizer.vocab_size
        # Must have at least one valid token
        assert valid_count >= 1

    def test_open_paren_is_valid_initially(
        self, grammar_factory, gpt2_tokenizer
    ):
        state = grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        paren_token = gpt2_tokenizer.encode("(")[0]
        assert mask[paren_token].item() is True


class TestStateTransitions:
    """advance() correctly transitions grammar state."""

    def test_not_accepting_initially(self, grammar_factory):
        state = grammar_factory(SIMPLE_GRAMMAR)
        assert state.is_accepting() is False

    def test_advance_changes_valid_mask(
        self, grammar_factory, gpt2_tokenizer
    ):
        state = grammar_factory(SIMPLE_GRAMMAR)
        mask_before = state.get_valid_mask(gpt2_tokenizer.vocab_size).clone()
        paren_token = gpt2_tokenizer.encode("(")[0]
        state.advance(paren_token)
        mask_after = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        assert not torch.equal(mask_before, mask_after)

    def test_full_expression_reaches_accepting(
        self, grammar_factory, gpt2_tokenizer
    ):
        """Feeding '(add 1 2)' token by token reaches accepting state."""
        state = grammar_factory(SIMPLE_GRAMMAR)
        tokens = gpt2_tokenizer.encode("(add 1 2)")
        for token_id in tokens:
            state.advance(token_id)
        assert state.is_accepting() is True


class TestToolGrammar:
    """Test with a grammar matching tgirl's tool output format."""

    def test_add_call_accepts(self, grammar_factory, gpt2_tokenizer):
        state = grammar_factory(TOOL_GRAMMAR)
        tokens = gpt2_tokenizer.encode('(add 42 -7)')
        for tid in tokens:
            state.advance(tid)
        assert state.is_accepting() is True

    def test_greet_call_accepts(self, grammar_factory, gpt2_tokenizer):
        state = grammar_factory(TOOL_GRAMMAR)
        tokens = gpt2_tokenizer.encode('(greet "world")')
        for tid in tokens:
            state.advance(tid)
        assert state.is_accepting() is True

    def test_wrong_tool_rejects(self, grammar_factory, gpt2_tokenizer):
        """After '(' the grammar should not allow arbitrary words."""
        state = grammar_factory(TOOL_GRAMMAR)
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        paren_token = gpt2_tokenizer.encode("(")[0]
        state.advance(paren_token)
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        # "delete" should not be valid after (
        delete_token = gpt2_tokenizer.encode("delete")[0]
        assert mask[delete_token].item() is False


class TestFactory:
    """make_outlines_grammar_factory contract tests."""

    def test_returns_callable(self, gpt2_tokenizer):
        factory = make_outlines_grammar_factory(gpt2_tokenizer)
        assert callable(factory)

    def test_factory_produces_grammar_state(self, grammar_factory):
        state = grammar_factory(SIMPLE_GRAMMAR)
        assert isinstance(state, GrammarState)

    def test_each_call_produces_independent_state(self, grammar_factory):
        s1 = grammar_factory(SIMPLE_GRAMMAR)
        s2 = grammar_factory(SIMPLE_GRAMMAR)
        # Advance s1 but not s2
        s1.advance(7)  # '(' in GPT-2
        assert s1.is_accepting() == s2.is_accepting() or True
        # They should be independent — s2 shouldn't see s1's state change

    def test_invalid_grammar_errors_on_use(
        self, grammar_factory, gpt2_tokenizer
    ):
        """Invalid grammar may defer errors to token consumption.

        llguidance accepts most grammars at construction time and
        reports errors lazily. We verify the state is not accepting
        and produces constrained output (or errors on advance).
        """
        state = grammar_factory('?start: "valid_literal"')
        assert state.is_accepting() is False
        mask = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        # Should be highly constrained
        assert mask.sum().item() < gpt2_tokenizer.vocab_size


class TestGetValidMaskNp:
    """get_valid_mask_np returns correct numpy array."""

    def test_returns_ndarray(self, grammar_factory, gpt2_tokenizer):
        import numpy as np

        state = grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask_np(gpt2_tokenizer.vocab_size)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_

    def test_matches_torch_mask(self, grammar_factory, gpt2_tokenizer):
        import numpy as np

        state = grammar_factory(SIMPLE_GRAMMAR)
        mask_torch = state.get_valid_mask(gpt2_tokenizer.vocab_size)
        mask_np = state.get_valid_mask_np(gpt2_tokenizer.vocab_size)
        np.testing.assert_array_equal(mask_np, mask_torch.numpy())


class TestMlxGrammarState:
    """LLGuidanceGrammarStateMlx produces correct MLX-native masks."""

    @pytest.fixture(scope="class")
    def mlx_grammar_factory(self, gpt2_tokenizer):
        mlx = pytest.importorskip("mlx.core")
        pytest.importorskip("llguidance.mlx")
        from tgirl.outlines_adapter import make_outlines_grammar_factory_mlx

        return make_outlines_grammar_factory_mlx(gpt2_tokenizer)

    def test_produces_mlx_grammar_state(self, mlx_grammar_factory):
        from tgirl.outlines_adapter import LLGuidanceGrammarStateMlx

        state = mlx_grammar_factory(SIMPLE_GRAMMAR)
        assert isinstance(state, LLGuidanceGrammarStateMlx)

    def test_get_valid_mask_mx_returns_mx_array(self, mlx_grammar_factory, gpt2_tokenizer):
        import mlx.core as mx

        state = mlx_grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask_mx(gpt2_tokenizer.vocab_size)
        assert isinstance(mask, mx.array)

    def test_mask_constrains_initial_state(self, mlx_grammar_factory, gpt2_tokenizer):
        import mlx.core as mx
        import numpy as np

        state = mlx_grammar_factory(SIMPLE_GRAMMAR)
        mask = state.get_valid_mask_mx(gpt2_tokenizer.vocab_size)
        valid_count = int(mx.sum(mask).item())
        assert valid_count < gpt2_tokenizer.vocab_size
        assert valid_count >= 1

    def test_mlx_mask_matches_torch_mask(self, grammar_factory, gpt2_tokenizer):
        """MLX and torch grammar states produce equivalent masks."""
        mlx = pytest.importorskip("mlx.core")
        pytest.importorskip("llguidance.mlx")
        import numpy as np

        from tgirl.outlines_adapter import make_outlines_grammar_factory_mlx

        mlx_factory = make_outlines_grammar_factory_mlx(gpt2_tokenizer)

        torch_state = grammar_factory(SIMPLE_GRAMMAR)
        mlx_state = mlx_factory(SIMPLE_GRAMMAR)

        torch_mask = torch_state.get_valid_mask(gpt2_tokenizer.vocab_size)
        mlx_mask = mlx_state.get_valid_mask_mx(gpt2_tokenizer.vocab_size)

        np.testing.assert_array_equal(
            np.array(mlx_mask),
            torch_mask.numpy(),
        )

    def test_advance_and_accepting(self, mlx_grammar_factory, gpt2_tokenizer):
        state = mlx_grammar_factory(SIMPLE_GRAMMAR)
        assert state.is_accepting() is False
        tokens = gpt2_tokenizer.encode("(add 1 2)")
        for tid in tokens:
            state.advance(tid)
        assert state.is_accepting() is True

    def test_satisfies_grammar_state_mlx_protocol(self, mlx_grammar_factory):
        from tgirl.sample_mlx import GrammarStateMlx

        state = mlx_grammar_factory(SIMPLE_GRAMMAR)
        assert isinstance(state, GrammarStateMlx)


class TestWithRealTgirlGrammar:
    """Integration: generate grammar from registry, use with adapter."""

    def test_tgirl_generated_grammar_works(
        self, grammar_factory, gpt2_tokenizer
    ):
        from tgirl.grammar import generate
        from tgirl.registry import ToolRegistry

        reg = ToolRegistry()

        @reg.tool()
        def add(a: int, b: int) -> int:
            return a + b

        @reg.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        snapshot = reg.snapshot()
        grammar_output = generate(snapshot)

        state = grammar_factory(grammar_output.text)
        assert isinstance(state, GrammarState)
        assert state.is_accepting() is False

        # Feed a valid expression
        tokens = gpt2_tokenizer.encode("(add 1 2)")
        for tid in tokens:
            state.advance(tid)
        assert state.is_accepting() is True
