"""Tests for SamplingSession + ToolRouter integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from tgirl.sample import GrammarState, SamplingSession
from tgirl.types import (
    ParameterDef,
    PrimitiveType,
    RerankConfig,
    SessionConfig,
    ToolDefinition,
)

# --- Helpers ---


def _make_registry_mock(tool_names: list[str]) -> MagicMock:
    """Create a mock ToolRegistry with given tools."""
    from tgirl.types import RegistrySnapshot

    tools = tuple(
        ToolDefinition(
            name=n,
            parameters=(ParameterDef(name="x", type_repr=PrimitiveType(kind="str")),),
            return_type=PrimitiveType(kind="str"),
            description=f"Tool {n}",
        )
        for n in tool_names
    )
    snap = RegistrySnapshot(
        tools=tools,
        quotas={n: 5 for n in tool_names},
        cost_remaining=None,
        scopes=frozenset(),
        timestamp=1000.0,
    )
    registry = MagicMock()
    registry.snapshot.return_value = snap
    registry.names.return_value = tool_names
    return registry


class TestSamplingSessionRerankConfig:
    """SamplingSession accepts rerank_config parameter."""

    def test_session_accepts_rerank_config_none(self) -> None:
        """Default behavior: no rerank_config."""
        registry = _make_registry_mock(["alpha", "beta"])
        session = SamplingSession(
            registry=registry,
            forward_fn=MagicMock(),
            tokenizer_decode=MagicMock(),
            tokenizer_encode=MagicMock(),
            embeddings=torch.randn(100, 32),
            grammar_guide_factory=MagicMock(),
            rerank_config=None,
        )
        assert session._rerank_config is None
        assert session._router is None

    def test_session_accepts_rerank_config(self) -> None:
        """When rerank_config is provided, ToolRouter is created."""
        registry = _make_registry_mock(["alpha", "beta"])
        config = RerankConfig(max_tokens=8)
        session = SamplingSession(
            registry=registry,
            forward_fn=MagicMock(),
            tokenizer_decode=MagicMock(),
            tokenizer_encode=MagicMock(),
            embeddings=torch.randn(100, 32),
            grammar_guide_factory=MagicMock(),
            rerank_config=config,
        )
        assert session._rerank_config is not None
        assert session._router is not None


class TestSamplingSessionRerankIntegration:
    """Verify ToolRouter is called during run() and snapshot is restricted."""

    def test_router_called_during_constrained_mode(self) -> None:
        """When rerank_config is active, ToolRouter.route() is called
        during constrained mode and the restricted snapshot is used."""
        from tgirl.grammar import GrammarOutput
        from tgirl.types import RerankResult

        registry = _make_registry_mock(["alpha", "beta"])
        rerank_config = RerankConfig(max_tokens=8)

        # Mock grammar state that accepts immediately
        mock_gs = MagicMock(spec=GrammarState)
        mock_gs.get_valid_mask.return_value = torch.ones(100, dtype=torch.bool)
        mock_gs.is_accepting.return_value = True

        factory = MagicMock(return_value=mock_gs)
        forward_fn = MagicMock(return_value=torch.randn(100))

        # tokenizer_decode: return delimiter chars to trigger detection
        delimiter = "<tool>"
        call_count = {"n": 0}

        def decode_fn(tokens: list[int]) -> str:
            call_count["n"] += 1
            if call_count["n"] <= len(delimiter):
                return delimiter[call_count["n"] - 1]
            return "(alpha 1)"

        encode_fn = MagicMock(return_value=[50, 51])
        embeddings = torch.randn(100, 32)

        session = SamplingSession(
            registry=registry,
            forward_fn=forward_fn,
            tokenizer_decode=decode_fn,
            tokenizer_encode=encode_fn,
            embeddings=embeddings,
            grammar_guide_factory=factory,
            rerank_config=rerank_config,
            config=SessionConfig(
                max_tool_cycles=1,
                freeform_max_tokens=20,
                session_timeout=10.0,
            ),
        )

        # Patch router.route to return a known result
        mock_rerank_result = RerankResult(
            selected_tools=("alpha",),
            routing_tokens=2,
            routing_latency_ms=5.0,
            routing_grammar_text='start: tool_choice\ntool_choice: "alpha"\n',
        )

        # Track what snapshot is passed to generate_grammar
        captured_snapshots: list = []

        def fake_generate(snapshot):
            captured_snapshots.append(snapshot)
            return GrammarOutput(
                text="start: expr",
                productions=(),
                snapshot_hash="test",
                tool_quotas={},
                cost_remaining=None,
            )

        with (
            patch.object(
                session._router, "route", return_value=mock_rerank_result
            ) as mock_route,
            patch("tgirl.compile.run_pipeline", return_value="ok"),
            patch("tgirl.grammar.generate", side_effect=fake_generate),
        ):
            result = session.run(prompt_tokens=[1, 2, 3])

            # ToolRouter.route() must have been called
            assert mock_route.call_count >= 1

            # The snapshot passed to generate_grammar should be restricted
            # to only the selected tool ("alpha")
            assert len(captured_snapshots) >= 1
            restricted_snap = captured_snapshots[0]
            tool_names = [t.name for t in restricted_snap.tools]
            assert "alpha" in tool_names
            assert "beta" not in tool_names

            # Routing tokens should be counted
            assert result.total_tokens >= mock_rerank_result.routing_tokens
