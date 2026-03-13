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
from hy.models import Expression, List, Object, Symbol
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


# Composition operators and special forms allowed in Hy AST
_COMPOSITION_KEYWORDS = frozenset({
    "->", "let", "if", "try", "except", "catch", "pmap",
    "insufficient-resources",
})

# Dangerous builtins that must never appear as call targets
_DANGEROUS_BUILTINS = frozenset({
    "__import__", "open", "getattr", "setattr", "delattr",
})

# Definition forms that are not allowed (no recursive definitions)
_DEFINITION_FORMS = frozenset({
    "defn", "defmacro", "defclass", "deftype",
})

# Import-like forms
_IMPORT_FORMS = frozenset({"import", "require"})


def _analyze_hy_ast(
    trees: list[Object], tool_names: set[str]
) -> PipelineError | None:
    """Walk the Hy AST and check for disallowed constructs.

    Returns None if analysis passes, PipelineError if any check fails.
    """
    # Combine tool names with composition keywords for call target validation
    allowed_calls = tool_names | _COMPOSITION_KEYWORDS

    # Track variables bound by let and threading
    bound_vars: set[str] = set()

    def _check_node(
        node: Object, in_let_bindings: bool = False
    ) -> PipelineError | None:
        if isinstance(node, Expression) and len(node) > 0:
            head = node[0]

            if isinstance(head, Symbol):
                name = str(head)

                # Check import/require forms
                if name in _IMPORT_FORMS:
                    return PipelineError(
                        stage=STAGE_STATIC_ANALYSIS,
                        error_type="DisallowedForm",
                        message=f"'{name}' form is not allowed",
                        hy_source=str(node),
                    )

                # Check definition forms
                if name in _DEFINITION_FORMS:
                    return PipelineError(
                        stage=STAGE_STATIC_ANALYSIS,
                        error_type="DisallowedForm",
                        message=f"'{name}' form is not allowed",
                        hy_source=str(node),
                    )

                # Check dangerous builtins
                if name in _DANGEROUS_BUILTINS:
                    return PipelineError(
                        stage=STAGE_STATIC_ANALYSIS,
                        error_type="DangerousBuiltin",
                        message=f"Dangerous builtin '{name}' is not allowed",
                        hy_source=str(node),
                    )

                # Attribute access: (. obj attr)
                if name == ".":
                    if len(node) >= 3:
                        attr = str(node[2])
                        if attr.startswith("__") and attr.endswith("__"):
                            return PipelineError(
                                stage=STAGE_STATIC_ANALYSIS,
                                error_type="DunderAccess",
                                message=(
                                    f"Dunder attribute '{attr}' access"
                                    " is not allowed"
                                ),
                                hy_source=str(node),
                            )
                    # Only recurse into the object (node[1]), not attribute names
                    if len(node) >= 2:
                        err = _check_node(node[1])
                        if err:
                            return err
                    return None

                # Let form: bind variables
                if name == "let":
                    if len(node) >= 2 and isinstance(node[1], List):
                        bindings = node[1]
                        # Bindings are pairs: [name1 val1 name2 val2 ...]
                        for i in range(0, len(bindings), 2):
                            if isinstance(bindings[i], Symbol):
                                bound_vars.add(str(bindings[i]))
                            # Check the value expression
                            if i + 1 < len(bindings):
                                err = _check_node(bindings[i + 1])
                                if err:
                                    return err
                    # Check body expressions
                    for child in node[2:]:
                        err = _check_node(child)
                        if err:
                            return err
                    return None

                # try/except: bind exception variable
                if name == "try":
                    for child in node[1:]:
                        if isinstance(child, Expression) and len(child) > 0:
                            child_head = child[0]
                            is_except = (
                                isinstance(child_head, Symbol)
                                and str(child_head) == "except"
                            )
                            if is_except:
                                # except [e Exception] body...
                                if len(child) >= 2 and isinstance(child[1], List):
                                    exc_bindings = child[1]
                                    has_exc_var = (
                                        len(exc_bindings) >= 1
                                        and isinstance(exc_bindings[0], Symbol)
                                    )
                                    if has_exc_var:
                                        bound_vars.add(str(exc_bindings[0]))
                                    # Exception type name is allowed
                                    has_exc_type = (
                                        len(exc_bindings) >= 2
                                        and isinstance(exc_bindings[1], Symbol)
                                    )
                                    if has_exc_type:
                                        bound_vars.add(str(exc_bindings[1]))
                                # Recurse into except body
                                for body_child in child[2:]:
                                    err = _check_node(body_child)
                                    if err:
                                        return err
                                continue
                        err = _check_node(child)
                        if err:
                            return err
                    return None

                # pmap: [fn1 fn2] arg — validate fn list contents
                if name == "pmap":
                    if len(node) >= 2 and isinstance(node[1], List):
                        for fn_sym in node[1]:
                            if isinstance(fn_sym, Symbol):
                                fn_name = str(fn_sym)
                                if fn_name not in tool_names:
                                    return PipelineError(
                                        stage=STAGE_STATIC_ANALYSIS,
                                        error_type="UnregisteredTool",
                                        message=(
                                        f"Function '{fn_name}' in pmap"
                                        " is not a registered tool"
                                    ),
                                        hy_source=str(node),
                                    )
                    # Check remaining args
                    for child in node[2:]:
                        err = _check_node(child)
                        if err:
                            return err
                    return None

                # Check if call target is allowed
                if name not in allowed_calls and name != ".":
                    return PipelineError(
                        stage=STAGE_STATIC_ANALYSIS,
                        error_type="UnregisteredTool",
                        message=(
                            f"Function '{name}' is not a registered"
                            " tool or composition operator"
                        ),
                        hy_source=str(node),
                    )

            # Recurse into sub-expressions
            for child in node[1:]:
                err = _check_node(child)
                if err:
                    return err

        elif isinstance(node, Symbol):
            name = str(node)
            # Allow: tool names, composition keywords, let-bound vars,
            # Python builtins used as values (True, False, None),
            # and string/number literals (handled by other model types)
            python_constants = {"True", "False", "None"}
            if (
                name not in tool_names
                and name not in _COMPOSITION_KEYWORDS
                and name not in bound_vars
                and name not in python_constants
                and not name.startswith("_hy_")  # Hy internal vars
            ):
                return PipelineError(
                    stage=STAGE_STATIC_ANALYSIS,
                    error_type="UnresolvedReference",
                    message=(
                    f"Symbol '{name}' is not a registered"
                    " tool, bound variable, or keyword"
                ),
                    hy_source=name,
                )

        elif isinstance(node, List):
            for child in node:
                err = _check_node(child)
                if err:
                    return err

        return None

    for tree in trees:
        err = _check_node(tree)
        if err:
            return err

    return None


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
