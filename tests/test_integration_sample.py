"""Integration tests for tgirl.sample — verify module boundaries work together."""

from __future__ import annotations


class TestSampleExports:
    """Task 8: All new public names are importable from tgirl."""

    def test_session_config_importable(self) -> None:
        from tgirl import SessionConfig

        assert SessionConfig is not None

    def test_model_intervention_importable(self) -> None:
        from tgirl import ModelIntervention

        assert ModelIntervention is not None

    def test_inference_hook_importable(self) -> None:
        from tgirl import InferenceHook

        assert InferenceHook is not None

    def test_grammar_state_importable(self) -> None:
        from tgirl import GrammarState

        assert GrammarState is not None

    def test_grammar_temperature_hook_importable(self) -> None:
        from tgirl import GrammarTemperatureHook

        assert GrammarTemperatureHook is not None

    def test_sampling_session_importable(self) -> None:
        from tgirl import SamplingSession

        assert SamplingSession is not None

    def test_sampling_result_importable(self) -> None:
        from tgirl import SamplingResult

        assert SamplingResult is not None

    def test_merge_interventions_importable(self) -> None:
        from tgirl import merge_interventions

        assert merge_interventions is not None

    def test_apply_penalties_importable(self) -> None:
        from tgirl import apply_penalties

        assert apply_penalties is not None

    def test_apply_shaping_importable(self) -> None:
        from tgirl import apply_shaping

        assert apply_shaping is not None

    def test_run_constrained_generation_importable(self) -> None:
        from tgirl import run_constrained_generation

        assert run_constrained_generation is not None


class TestSampleIntegration:
    """Task 8: Integration test exercising the full pipeline with mocks."""

    def test_constrained_generation_with_transport(self) -> None:
        """End-to-end: registry -> grammar -> constrained generation -> result."""
        import torch

        from tgirl.registry import ToolRegistry
        from tgirl.sample import run_constrained_generation
        from tgirl.transport import TransportConfig

        # Register a tool
        registry = ToolRegistry()

        @registry.tool(quota=3, cost=1.0)
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Create a mock grammar state that accepts after 2 tokens
        class MockGS:
            def __init__(self) -> None:
                self._advances = 0

            def get_valid_mask(self, vocab_size: int) -> torch.Tensor:
                mask = torch.zeros(vocab_size, dtype=torch.bool)
                # Allow a few tokens
                mask[:10] = True
                return mask

            def is_accepting(self) -> bool:
                return self._advances >= 2

            def advance(self, token_id: int) -> None:
                self._advances += 1

        vocab_size = 50
        embeddings = torch.eye(vocab_size)

        result = run_constrained_generation(
            grammar_state=MockGS(),
            forward_fn=lambda ctx: torch.randn(vocab_size),
            tokenizer_decode=lambda ids: "".join(chr(65 + (i % 26)) for i in ids),
            embeddings=embeddings,
            hooks=[],
            transport_config=TransportConfig(),
            max_tokens=10,
        )

        # Verify result structure
        assert len(result.tokens) == 2
        assert len(result.grammar_valid_counts) == 2
        assert all(c == 10 for c in result.grammar_valid_counts)
        assert len(result.wasserstein_distances) == 2
        assert result.ot_computation_total_ms >= 0
        assert result.hy_source  # Non-empty decoded string

    def test_ruff_passes_on_sample_module(self) -> None:
        """Verify ruff check passes on sample.py and types.py."""
        import subprocess

        proc = subprocess.run(
            ["ruff", "check", "src/tgirl/sample.py", "src/tgirl/types.py"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"ruff check failed:\n{proc.stdout}\n{proc.stderr}"
