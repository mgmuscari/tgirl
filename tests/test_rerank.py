"""Tests for tgirl.rerank — ToolRouter for grammar-constrained re-ranking."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from tgirl.sample import ConstrainedGenerationResult, GrammarState
from tgirl.state_machine import Checkpoint  # noqa: F401 — triggers model_rebuild

# Resolve Checkpoint forward ref so ConstrainedGenerationResult can be instantiated
ConstrainedGenerationResult.model_rebuild()

from tgirl.types import (
    ParameterDef,
    PrimitiveType,
    RegistrySnapshot,
    RerankConfig,
    RerankResult,
    ToolDefinition,
)


# --- Helpers ---


def _make_tool(name: str, *, description: str = "") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        parameters=(
            ParameterDef(name="x", type_repr=PrimitiveType(kind="str")),
        ),
        return_type=PrimitiveType(kind="str"),
        description=description or f"Tool {name}",
    )


def _make_snapshot(
    tool_names: list[str],
    quotas: dict[str, int] | None = None,
) -> RegistrySnapshot:
    tools = tuple(_make_tool(n) for n in tool_names)
    q = quotas or {}
    return RegistrySnapshot(
        tools=tools,
        quotas=q,
        cost_remaining=None,
        scopes=frozenset(),
        timestamp=time.time(),
    )


def _make_mock_grammar_state() -> MagicMock:
    """Create a mock GrammarState that accepts after first advance."""
    gs = MagicMock(spec=GrammarState)
    gs.get_valid_mask.return_value = torch.ones(100, dtype=torch.bool)
    gs.is_accepting.return_value = True
    return gs


class TestToolRouterSingleTool:
    """Single-tool snapshot returns that tool immediately (no generation)."""

    def test_single_tool_short_circuits(self) -> None:
        from tgirl.rerank import ToolRouter

        factory = MagicMock()
        forward_fn = MagicMock()
        decode = MagicMock()
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["only_tool"])
        result = router.route(snap, context_tokens=[1, 2, 3])

        assert result.selected_tools == ("only_tool",)
        assert result.routing_tokens == 0
        # No generation should have been called
        forward_fn.assert_not_called()


class TestToolRouterEmptySnapshot:
    """Empty snapshot raises ValueError."""

    def test_empty_snapshot_raises(self) -> None:
        from tgirl.rerank import ToolRouter

        router = ToolRouter(
            grammar_guide_factory=MagicMock(),
            forward_fn=MagicMock(),
            tokenizer_decode=MagicMock(),
            embeddings=torch.randn(100, 32),
        )
        snap = _make_snapshot([])
        with pytest.raises(ValueError, match="empty snapshot"):
            router.route(snap, context_tokens=[1, 2, 3])


class TestToolRouterRoute:
    """Tests for the main route() method."""

    def test_route_returns_rerank_result(self) -> None:
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="get_field")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["get_field", "set_field"])
        result = router.route(snap, context_tokens=[1, 2, 3])

        assert isinstance(result, RerankResult)
        assert "get_field" in result.selected_tools
        assert result.routing_tokens >= 0

    def test_route_calls_run_constrained_generation_with_empty_hooks(
        self,
    ) -> None:
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            router.route(snap, context_tokens=[1, 2])

            # Verify hooks=[] was passed
            _, kwargs = mock_gen.call_args
            assert kwargs["hooks"] == []

    def test_config_enabled_false_returns_all_tools(self) -> None:
        from tgirl.rerank import ToolRouter

        config = RerankConfig(enabled=False)
        router = ToolRouter(
            grammar_guide_factory=MagicMock(),
            forward_fn=MagicMock(),
            tokenizer_decode=MagicMock(),
            embeddings=torch.randn(100, 32),
            config=config,
        )
        snap = _make_snapshot(["alpha", "beta", "gamma"])
        result = router.route(snap, context_tokens=[1, 2])

        assert set(result.selected_tools) == {"alpha", "beta", "gamma"}
        assert result.routing_tokens == 0


class TestToolRouterQuotaFiltering:
    """Tests for quota-exhausted tool filtering."""

    def test_exhausted_tools_excluded_from_routing(self) -> None:
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="beta")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        # alpha has quota=0 (exhausted), beta has quota=3
        snap = _make_snapshot(
            ["alpha", "beta"],
            quotas={"alpha": 0, "beta": 3},
        )
        result = router.route(snap, context_tokens=[1, 2])

        # alpha is exhausted, only beta remains -> short-circuit
        assert result.selected_tools == ("beta",)
        assert result.routing_tokens == 0  # short-circuit, no generation

    def test_all_tools_exhausted_raises(self) -> None:
        from tgirl.rerank import ToolRouter

        router = ToolRouter(
            grammar_guide_factory=MagicMock(),
            forward_fn=MagicMock(),
            tokenizer_decode=MagicMock(),
            embeddings=torch.randn(100, 32),
        )
        snap = _make_snapshot(
            ["alpha", "beta"],
            quotas={"alpha": 0, "beta": 0},
        )
        with pytest.raises(ValueError, match="empty snapshot"):
            router.route(snap, context_tokens=[1, 2])

    def test_single_remaining_after_quota_filter_short_circuits(self) -> None:
        from tgirl.rerank import ToolRouter

        forward_fn = MagicMock()
        router = ToolRouter(
            grammar_guide_factory=MagicMock(),
            forward_fn=forward_fn,
            tokenizer_decode=MagicMock(),
            embeddings=torch.randn(100, 32),
        )
        snap = _make_snapshot(
            ["alpha", "beta", "gamma"],
            quotas={"alpha": 0, "beta": 0, "gamma": 2},
        )
        result = router.route(snap, context_tokens=[1])

        assert result.selected_tools == ("gamma",)
        forward_fn.assert_not_called()


class TestToolRouterCache:
    """Tests for routing grammar compilation caching."""

    def test_cache_hit_reuses_grammar_text(self) -> None:
        """Cache stores grammar text; generate_routing_grammar is not called
        on cache hit. Factory is still called each time for fresh GrammarState
        (since GrammarState is mutated by advance())."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen, \
             patch("tgirl.rerank.generate_routing_grammar") as mock_gen_grammar:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen_grammar.return_value = (
                'start: tool_choice\ntool_choice: "alpha" | "beta"\n'
            )
            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            router.route(snap, context_tokens=[1, 2])
            router.route(snap, context_tokens=[1, 2])

            # generate_routing_grammar called once, cached on second
            assert mock_gen_grammar.call_count == 1
            # Factory called twice (fresh GrammarState each time)
            assert factory.call_count == 2

    def test_cache_invalidates_on_tool_set_change(self) -> None:
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            snap1 = _make_snapshot(["alpha", "beta"])
            snap2 = _make_snapshot(["alpha", "gamma"])

            router.route(snap1, context_tokens=[1])
            router.route(snap2, context_tokens=[1])

            # Factory called twice (different tool sets)
            assert factory.call_count == 2


class TestToolRouterContextPassthrough:
    """Tests that context_tokens are passed through as-is (no prompt prepending)."""

    def test_context_tokens_passed_through_unchanged(self) -> None:
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            router.route(snap, context_tokens=[1, 2, 3])

            # context_tokens should be passed through as-is, no prepending
            _, kwargs = mock_gen.call_args
            assert kwargs["context_tokens"] == [1, 2, 3]


class TestToolRouterValidation:
    """Tests for output validation."""

    def test_invalid_tool_name_raises(self) -> None:
        """If generation produces a name not in the tool set, raise ValueError."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="bogus_tool")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["alpha", "beta"])

        with pytest.raises(ValueError, match="contains no valid tools"):
            router.route(snap, context_tokens=[1, 2])


class TestToolRouterMultiTool:
    """Tests for multi-tool routing with top_k > 1."""

    def test_multi_tool_output_parsed(self) -> None:
        """Router parses space-separated tool names when top_k > 1."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha beta")
        embeddings = torch.randn(100, 32)

        config = RerankConfig(top_k=3)
        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            config=config,
        )
        snap = _make_snapshot(["alpha", "beta", "gamma"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5, 6],
                hy_source="alpha beta",
                grammar_valid_counts=[3, 3],
                temperatures_applied=[0.3, 0.3],
                wasserstein_distances=[0.0, 0.0],
                top_p_applied=[-1.0, -1.0],
                token_log_probs=[-0.5, -0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None, None],
                ot_iterations=[5, 5],
                grammar_generation_ms=5.0,
            )
            result = router.route(snap, context_tokens=[1, 2])

        assert result.selected_tools == ("alpha", "beta")

    def test_multi_tool_deduplicates(self) -> None:
        """Duplicate tool names in output are deduplicated, preserving order."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha alpha beta")
        embeddings = torch.randn(100, 32)

        config = RerankConfig(top_k=3)
        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            config=config,
        )
        snap = _make_snapshot(["alpha", "beta", "gamma"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5, 6, 7],
                hy_source="alpha alpha beta",
                grammar_valid_counts=[3, 3, 3],
                temperatures_applied=[0.3, 0.3, 0.3],
                wasserstein_distances=[0.0, 0.0, 0.0],
                top_p_applied=[-1.0, -1.0, -1.0],
                token_log_probs=[-0.5, -0.5, -0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None, None, None],
                ot_iterations=[5, 5, 5],
                grammar_generation_ms=5.0,
            )
            result = router.route(snap, context_tokens=[1, 2])

        assert result.selected_tools == ("alpha", "beta")

    def test_multi_tool_respects_top_k(self) -> None:
        """Result is capped at top_k even if model outputs more."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha beta gamma")
        embeddings = torch.randn(100, 32)

        config = RerankConfig(top_k=2)
        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            config=config,
        )
        snap = _make_snapshot(["alpha", "beta", "gamma"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5, 6, 7],
                hy_source="alpha beta gamma",
                grammar_valid_counts=[3, 3, 3],
                temperatures_applied=[0.3, 0.3, 0.3],
                wasserstein_distances=[0.0, 0.0, 0.0],
                top_p_applied=[-1.0, -1.0, -1.0],
                token_log_probs=[-0.5, -0.5, -0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None, None, None],
                ot_iterations=[5, 5, 5],
                grammar_generation_ms=5.0,
            )
            result = router.route(snap, context_tokens=[1, 2])

        assert len(result.selected_tools) == 2
        assert result.selected_tools == ("alpha", "beta")

    def test_top_k_1_backwards_compat(self) -> None:
        """top_k=1 (default) produces single-element tuple as before."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            result = router.route(snap, context_tokens=[1, 2])

        assert result.selected_tools == ("alpha",)

    def test_multi_tool_passes_top_k_to_grammar(self) -> None:
        """Router passes top_k to generate_routing_grammar."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        config = RerankConfig(top_k=3)
        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            config=config,
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch("tgirl.rerank.run_constrained_generation") as mock_gen, \
             patch("tgirl.rerank.generate_routing_grammar") as mock_grammar:
            from tgirl.sample import ConstrainedGenerationResult

            mock_grammar.return_value = (
                'start: tool_list\n'
                'tool_list: tool_choice (" " tool_choice)~0..2\n'
                'tool_choice: "alpha" | "beta"\n'
            )
            mock_gen.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            router.route(snap, context_tokens=[1, 2])

            # Verify top_k was passed to generate_routing_grammar
            mock_grammar.assert_called_once()
            _, kwargs = mock_grammar.call_args
            assert kwargs.get("top_k") == 3 or (
                mock_grammar.call_args[0]
                and len(mock_grammar.call_args[0]) > 1
                and mock_grammar.call_args[0][1] == 3
            )


class TestToolRouterMlxPath:
    """Tests for MLX backend dispatch in ToolRouter."""

    def test_router_torch_path_unchanged(self) -> None:
        """Torch backend router works as before."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            backend="torch",
        )
        snap = _make_snapshot(["alpha", "beta"])
        result = router.route(snap, context_tokens=[1, 2])

        assert "alpha" in result.selected_tools

    def test_router_mlx_path(self) -> None:
        """MLX backend router dispatches to run_constrained_generation_mlx."""
        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            backend="mlx",
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch(
            "tgirl.sample_mlx.run_constrained_generation_mlx"
        ) as mock_gen_mlx:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen_mlx.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            result = router.route(snap, context_tokens=[1, 2])

            mock_gen_mlx.assert_called_once()
            assert "alpha" in result.selected_tools

    def test_mlx_backend_uses_mlx_grammar_state(self) -> None:
        """Regression: when backend='mlx' and an mlx_grammar_guide_factory
        is configured, ToolRouter must use the MLX factory (returns
        GrammarStateMlx-compatible) — never the torch factory passed
        through a cross-framework conversion fallback.

        This locks out the CLAUDE.md violation (2026-03-14 gotcha) where
        sample_mlx.py used to convert torch valid_mask via
        mx.array(valid_mask_torch.numpy()) when the grammar state lacked
        get_valid_mask_mx.
        """
        from tgirl.rerank import ToolRouter

        # Torch factory that should NOT be called on the mlx path
        mock_torch_gs = _make_mock_grammar_state()
        torch_factory = MagicMock(return_value=mock_torch_gs)

        # MLX factory whose return has the get_valid_mask_mx method
        mock_mlx_gs = MagicMock()
        mock_mlx_gs.get_valid_mask_mx = MagicMock()
        mock_mlx_gs.is_accepting = MagicMock(return_value=True)
        mock_mlx_gs.advance = MagicMock()
        mlx_factory = MagicMock(return_value=mock_mlx_gs)

        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=torch_factory,
            mlx_grammar_guide_factory=mlx_factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            backend="mlx",
        )
        snap = _make_snapshot(["alpha", "beta"])

        with patch(
            "tgirl.sample_mlx.run_constrained_generation_mlx"
        ) as mock_gen_mlx:
            mock_gen_mlx.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            router.route(snap, context_tokens=[1, 2])

            # MLX factory used; torch factory not consulted
            mlx_factory.assert_called_once()
            torch_factory.assert_not_called()
            # The grammar_state passed to run_constrained_generation_mlx
            # is the one from the MLX factory.
            call_kwargs = mock_gen_mlx.call_args.kwargs
            assert call_kwargs["grammar_state"] is mock_mlx_gs

    def test_router_mlx_embeddings_lazy_conversion(self) -> None:
        """Torch embeddings converted to mx.array on first MLX route."""
        import mlx.core as mx

        from tgirl.rerank import ToolRouter

        mock_gs = _make_mock_grammar_state()
        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))
        decode = MagicMock(return_value="alpha")
        embeddings = torch.randn(100, 32)

        router = ToolRouter(
            grammar_guide_factory=factory,
            forward_fn=forward_fn,
            tokenizer_decode=decode,
            embeddings=embeddings,
            backend="mlx",
        )

        # Before first route, _embeddings_mlx should be None
        assert router._embeddings_mlx is None

        snap = _make_snapshot(["alpha", "beta"])

        with patch(
            "tgirl.sample_mlx.run_constrained_generation_mlx"
        ) as mock_gen_mlx:
            from tgirl.sample import ConstrainedGenerationResult

            mock_gen_mlx.return_value = ConstrainedGenerationResult(
                tokens=[5],
                hy_source="alpha",
                grammar_valid_counts=[2],
                temperatures_applied=[0.3],
                wasserstein_distances=[0.0],
                top_p_applied=[-1.0],
                token_log_probs=[-0.5],
                ot_computation_total_ms=1.0,
                ot_bypassed_count=0,
                ot_bypass_reasons=[None],
                ot_iterations=[5],
                grammar_generation_ms=5.0,
            )
            router.route(snap, context_tokens=[1, 2])

        # After first route, should have mx.array embeddings
        assert isinstance(router._embeddings_mlx, mx.array)
