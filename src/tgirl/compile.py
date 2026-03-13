"""Hy compilation and sandboxed execution for tgirl pipelines.

Parses Hy s-expressions, applies static analysis (Hy AST + Python AST),
and runs code in a sandboxed namespace containing only registered tool
callables and composition operator implementations.

Three-layer defense-in-depth:
1. Grammar prevents invalid expressions at token level
2. Static analysis catches template bugs before execution
3. Sandbox restricts the runtime namespace
"""

from __future__ import annotations

import re
from typing import Any

import hy
import structlog
from hy.models import Object
from pydantic import BaseModel, ConfigDict

from tgirl.registry import ToolRegistry
from tgirl.types import PipelineError

logger = structlog.get_logger()

# Pipeline stage constants
STAGE_PARSE = "parse"
STAGE_STATIC_ANALYSIS = "static_analysis"
STAGE_COMPILE = "compile"
STAGE_AST_ANALYSIS = "ast_analysis"
STAGE_EXECUTE = "execute"


class PipelineResult(BaseModel):
    """Successful pipeline execution result."""

    model_config = ConfigDict(frozen=True)
    result: Any
    hy_source: str
    execution_time_ms: float


class InsufficientResources(BaseModel):
    """Model's intentional signal that it cannot act.

    Not an error — represents a valid grammar alternative alongside
    pipeline and single_call (per TGIRL.md line 287).
    """

    model_config = ConfigDict(frozen=True)
    reason: str
    hy_source: str


class CompileConfig(BaseModel):
    """Configuration for the compile pipeline."""

    model_config = ConfigDict(frozen=True)
    pipeline_timeout: float = 60.0
    max_depth: int = 50



def _normalize_hy_source(source: str) -> str:
    """Normalize TGIRL spec forms to Hy-native forms.

    The TGIRL spec uses ``catch`` for error handling, but Hy 1.x
    uses ``except``.  Replaces the ``(catch`` form with ``(except``
    before Hy parsing.
    """
    return re.sub(r"\(catch\b", "(except", source)


def _parse_hy(source: str) -> list[Object] | PipelineError:
    """Parse a Hy source string into a list of Hy model objects.

    Normalizes spec-grammar forms (catch -> except) before parsing.
    Returns a list of Hy AST nodes on success, or PipelineError on failure.
    """
    if not source or not source.strip():
        return PipelineError(
            stage=STAGE_PARSE,
            error_type="EmptyInput",
            message="Empty source string",
            hy_source=source,
        )

    normalized = _normalize_hy_source(source)

    try:
        trees = list(hy.read_many(normalized))
    except Exception as exc:
        logger.warning("hy_parse_failed", source=source, error=str(exc))
        return PipelineError(
            stage=STAGE_PARSE,
            error_type=type(exc).__name__,
            message=str(exc),
            hy_source=source,
        )

    if not trees:
        return PipelineError(
            stage=STAGE_PARSE,
            error_type="EmptyInput",
            message="No expressions found in source",
            hy_source=source,
        )

    return trees


def run_pipeline(
    source: str,
    registry: ToolRegistry,
    config: CompileConfig | None = None,
) -> PipelineResult | InsufficientResources | PipelineError:
    """Execute a Hy s-expression pipeline.

    Stub implementation — returns PipelineError until fully implemented.
    """
    return PipelineError(
        stage=STAGE_EXECUTE,
        error_type="NotImplementedError",
        message="Pipeline not yet implemented",
        hy_source=source,
    )
