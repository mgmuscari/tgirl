"""Tests for SamplingSession + ToolRouter integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
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
            parameters=(
                ParameterDef(name="x", type_repr=PrimitiveType(kind="str")),
            ),
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
