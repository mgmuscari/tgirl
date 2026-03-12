"""Dynamic CFG generation from registry state.

Converts RegistrySnapshot objects into context-free grammars in Lark EBNF
format that constrain LLM token output to only produce well-formed Hy
s-expressions invoking registered tools.
"""

from __future__ import annotations

from collections.abc import Mapping

import structlog
from pydantic import BaseModel, ConfigDict

from tgirl.types import RegistrySnapshot

logger = structlog.get_logger()


class Production(BaseModel):
    """A single grammar production rule."""

    model_config = ConfigDict(frozen=True)
    name: str
    rule: str


class GrammarOutput(BaseModel):
    """Complete generated grammar with metadata."""

    model_config = ConfigDict(frozen=True)
    text: str
    productions: tuple[Production, ...]
    snapshot_hash: str
    tool_quotas: Mapping[str, int]
    cost_remaining: float | None


class GrammarDiff(BaseModel):
    """Diff between two generated grammars."""

    model_config = ConfigDict(frozen=True)
    added: tuple[Production, ...]
    removed: tuple[Production, ...]
    changed: tuple[tuple[Production, Production], ...]


class GrammarConfig(BaseModel):
    """Configuration for grammar generation."""

    model_config = ConfigDict(frozen=True)
    enumeration_threshold: int = 256


def generate(
    snapshot: RegistrySnapshot,
    config: GrammarConfig | None = None,
) -> GrammarOutput:
    """Generate a grammar from a registry snapshot.

    Args:
        snapshot: Immutable registry snapshot.
        config: Grammar generation configuration.

    Returns:
        Complete grammar output with text and metadata.

    Raises:
        NotImplementedError: Stub — not yet implemented.
    """
    raise NotImplementedError


def diff(a: GrammarOutput, b: GrammarOutput) -> GrammarDiff:
    """Compute the diff between two grammars.

    Args:
        a: First grammar.
        b: Second grammar.

    Returns:
        Diff showing added, removed, and changed productions.

    Raises:
        NotImplementedError: Stub — not yet implemented.
    """
    raise NotImplementedError
