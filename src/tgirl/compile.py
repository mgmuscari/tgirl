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

import ast
import concurrent.futures
import functools
import re
import time
from typing import Any

import hy
import structlog
from hy.compiler import hy_compile
from hy.models import Expression, List, Object, Symbol
from pydantic import BaseModel, ConfigDict
from RestrictedPython import RestrictingNodeTransformer

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


# --- Composition operator implementations ---


def _pmap_impl(fns: list, arg: Any) -> list:
    """Apply each function in fns to arg, return list of results.

    v1.0: sequential execution (fail-fast on first error).
    Per-tool timeouts are already applied via sandbox wrapping.
    """
    results = []
    for fn in fns:
        results.append(fn(arg))
    return results


def _insufficient_resources_impl(reason: str) -> InsufficientResources:
    """Return an InsufficientResources signal.

    This is the model's intentional signal that it cannot act,
    not an error condition.
    """
    return InsufficientResources(
        reason=reason, hy_source="(insufficient-resources ...)"
    )


def _expand_macros(tree: Object) -> Object:
    """Expand tgirl-specific macros in the Hy AST.

    Currently handles:
    - ``->`` (thread-first): ``(-> x (f) (g a))`` becomes ``(g (f x) a)``

    Hy 1.2 does not expand ``->`` as a macro during compilation,
    so we do it manually before calling hy_compile.
    """
    if not isinstance(tree, Expression) or len(tree) < 2:
        return tree

    head = tree[0]

    if isinstance(head, Symbol) and str(head) == "->":
        # Thread-first expansion
        result = _expand_macros(tree[1])
        for form in tree[2:]:
            form = _expand_macros(form)
            if isinstance(form, Expression):
                # (fn arg1 ...) -> (fn result arg1 ...)
                result = Expression(
                    [form[0], result, *list(form[1:])]
                )
            elif isinstance(form, Symbol):
                # bare symbol -> (fn result)
                result = Expression([form, result])
            else:
                result = Expression([form, result])
        return result

    # Recurse into sub-expressions
    return Expression(
        [
            _expand_macros(x) if isinstance(x, Expression) else x
            for x in tree
        ]
    )


# Composition operators and special forms allowed in Hy AST
_COMPOSITION_KEYWORDS = frozenset({
    "->", "let", "if", "try", "except", "catch", "pmap",
    "insufficient-resources",
})

# Dangerous builtins that must never appear as call targets
_DANGEROUS_BUILTINS = frozenset({
    "__import__", "open", "getattr", "setattr", "delattr",
})

# All disallowed forms: definitions, imports, and compile-time metaprogramming.
# These must be blocked before hy.compile() — macros and eval-* forms execute
# at compile time, bypassing the Python AST analyzer and sandbox entirely.
_DISALLOWED_FORMS = frozenset({
    # Definition forms (no recursive definitions)
    "defn", "defmacro", "defmacro/g!", "defclass", "deftype",
    # Import-like forms
    "import", "require", "include",
    # Compile-time metaprogramming (macro-expansion trap)
    "eval-and-compile", "eval-when-compile",
})


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

                # Check disallowed forms (definitions, imports,
                # compile-time metaprogramming)
                if name in _DISALLOWED_FORMS:
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
                    if len(node) < 2 or not isinstance(
                        node[1], List
                    ):
                        return PipelineError(
                            stage=STAGE_STATIC_ANALYSIS,
                            error_type="MalformedPmap",
                            message=(
                                "pmap requires a list of"
                                " functions as first argument"
                            ),
                            hy_source=str(node),
                        )
                    for fn_sym in node[1]:
                        if isinstance(fn_sym, Symbol):
                            fn_name = str(fn_sym)
                            if fn_name not in tool_names:
                                return PipelineError(
                                    stage=STAGE_STATIC_ANALYSIS,
                                    error_type="UnregisteredTool",
                                    message=(
                                        f"Function '{fn_name}'"
                                        " in pmap is not a"
                                        " registered tool"
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

            elif isinstance(head, Expression):
                # Expression-headed calls like (.upper "hello")
                # which Hy desugars to ((. None upper) "hello").
                # This is the method call bypass — reject it.
                return PipelineError(
                    stage=STAGE_STATIC_ANALYSIS,
                    error_type="DisallowedForm",
                    message=(
                        "Method call syntax is not allowed"
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


class _TgirlNodeTransformer(RestrictingNodeTransformer):
    """Subclass of RestrictingNodeTransformer for tgirl.

    Overrides:
    - Rejects Import/ImportFrom nodes
    - Rejects Global/Nonlocal nodes
    - Allows non-dunder attribute access without _getattr_ wrapping
    - Keeps dunder attribute rejection from parent
    """

    def visit_Import(  # noqa: N802
        self, node: ast.Import
    ) -> ast.Import:
        self.errors.append(
            f"Line {node.lineno}: Import statements are not allowed"
        )
        return node

    def visit_ImportFrom(  # noqa: N802
        self, node: ast.ImportFrom
    ) -> ast.ImportFrom:
        self.errors.append(
            f"Line {node.lineno}: Import statements are not allowed"
        )
        return node

    def visit_Global(  # noqa: N802
        self, node: ast.Global
    ) -> ast.Global:
        self.errors.append(
            f"Line {node.lineno}: Global statements are not allowed"
        )
        return node

    def visit_Nonlocal(  # noqa: N802
        self, node: ast.Nonlocal
    ) -> ast.Nonlocal:
        self.errors.append(
            f"Line {node.lineno}: Nonlocal statements are not"
            " allowed"
        )
        return node

    def check_name(
        self,
        node: ast.AST,
        name: str,
        allow_magic_methods: bool = False,
    ) -> None:
        """Allow Hy-internal variable names (_hy_*)."""
        if name is not None and name.startswith("_hy_"):
            return
        super().check_name(
            node, name, allow_magic_methods=allow_magic_methods
        )

    def visit_Attribute(  # noqa: N802
        self, node: ast.Attribute
    ) -> ast.AST:
        """Allow non-dunder attribute access, reject dunders."""
        attr_name = node.attr
        if attr_name.startswith("_"):
            self.errors.append(
                f"Line {node.lineno}: "
                f'"{attr_name}" is an invalid attribute name'
                ' because it starts with "_".'
            )
            return node
        # Allow non-dunder access without _getattr_ wrapping
        self.node_contents_visit(node)
        return node


def _analyze_python_ast(
    tree: ast.Module, tool_names: set[str]
) -> PipelineError | None:
    """Run RestrictedPython analysis on a Python AST.

    Uses a tgirl-specific subclass of RestrictingNodeTransformer
    that rejects imports, global/nonlocal, and dunder attributes
    while allowing non-dunder attribute access.
    """
    errors: list[str] = []
    transformer = _TgirlNodeTransformer(
        errors=errors, warnings=[], used_names={}
    )

    try:
        transformer.visit(tree)
    except Exception as exc:
        return PipelineError(
            stage=STAGE_AST_ANALYSIS,
            error_type=type(exc).__name__,
            message=str(exc),
            hy_source="<compiled AST>",
        )

    if errors:
        return PipelineError(
            stage=STAGE_AST_ANALYSIS,
            error_type="RestrictedPythonError",
            message="; ".join(errors),
            hy_source="<compiled AST>",
        )

    return None


def _run_with_timeout(
    fn: Any, timeout: float
) -> Any | PipelineError:
    """Run a callable with a timeout using ThreadPoolExecutor.

    Returns the result on success, or PipelineError on timeout.
    Uses manual executor management to avoid shutdown(wait=True)
    blocking the caller when the submitted task outlives the timeout.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return PipelineError(
            stage=STAGE_EXECUTE,
            error_type="TimeoutError",
            message=(
                f"Execution timed out after {timeout}s"
            ),
            hy_source="<timeout>",
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _wrap_with_timeout(
    fn: Any, timeout: float
) -> Any:
    """Wrap a callable with per-tool timeout enforcement.

    The wrapper raises TimeoutError if the call exceeds the timeout.
    Uses manual executor management to avoid shutdown(wait=True)
    blocking the caller.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            msg = (
                f"Tool '{fn.__name__}' timed out"
                f" after {timeout}s"
            )
            raise TimeoutError(msg) from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return wrapper


def _build_sandbox(registry: ToolRegistry) -> dict[str, Any]:
    """Build a restricted namespace for sandboxed execution.

    Contains only:
    - Registered tool callables
    - pmap and insufficient_resources implementations
    - _tgirl_result_ sentinel for result capture
    - Empty __builtins__ to prevent builtins access
    """
    sandbox: dict[str, Any] = {}

    # Add registered tool callables
    for name in registry.names():
        sandbox[name] = registry.get_callable(name)

    # Add composition operator implementations
    sandbox["pmap"] = _pmap_impl
    sandbox["insufficient_resources"] = _insufficient_resources_impl

    # Result capture sentinel
    sandbox["_tgirl_result_"] = None

    # Restricted builtins: only safe types needed by Hy-generated code
    sandbox["__builtins__"] = {
        "Exception": Exception,
        "True": True,
        "False": False,
        "None": None,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "dict": dict,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "tuple": tuple,
        "range": range,
    }

    return sandbox


def _inject_result_capture(tree: ast.Module) -> ast.Module:
    """Rewrite the last statement to capture its result value.

    For ``ast.Expr``: rewrite to ``_tgirl_result_ = <expr>``.
    For ``ast.Try``: Hy assigns to ``_hy_anon_var_*``; append
    ``_tgirl_result_ = _hy_anon_var_*`` after the try block.

    This runs AFTER security analysis — the injected assignment
    is trusted code from the engine, not model output.
    """
    if not tree.body:
        return tree

    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        assign = ast.Assign(
            targets=[
                ast.Name(id="_tgirl_result_", ctx=ast.Store())
            ],
            value=last.value,
        )
        ast.copy_location(assign, last)
        tree.body[-1] = assign
    elif isinstance(last, ast.Try):
        # Hy try/except assigns result to _hy_anon_var_*
        anon_var = _find_hy_anon_var(last)
        if anon_var:
            assign = ast.Assign(
                targets=[
                    ast.Name(
                        id="_tgirl_result_", ctx=ast.Store()
                    )
                ],
                value=ast.Name(
                    id=anon_var, ctx=ast.Load()
                ),
            )
            ast.copy_location(assign, last)
            tree.body.append(assign)

    ast.fix_missing_locations(tree)
    return tree


def _find_hy_anon_var(node: ast.AST) -> str | None:
    """Find a _hy_anon_var_* assignment target in an AST node."""
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.startswith("_hy_anon_var_")
                ):
                    return target.id
    return None


def run_pipeline(
    source: str,
    registry: ToolRegistry,
    config: CompileConfig | None = None,
) -> PipelineResult | InsufficientResources | PipelineError:
    """Execute a Hy s-expression pipeline.

    Pipeline stages:
    1. Parse Hy source
    2. Expand macros (-> threading)
    3. Static analysis (Hy AST)
    4. Compile to Python AST
    5. Security analysis (RestrictedPython)
    6. Inject result capture
    7. Build sandbox
    8. Execute with timeout
    """
    cfg = config or CompileConfig()
    start_time = time.monotonic()
    tool_names = set(registry.names())

    # Stage 1: Parse
    logger.debug("pipeline_parse", source=source)
    trees = _parse_hy(source)
    if isinstance(trees, PipelineError):
        return trees

    # Stage 1.5: Expand macros (-> threading)
    trees = [_expand_macros(t) for t in trees]

    # Stage 2: Hy AST static analysis
    logger.debug("pipeline_static_analysis")
    hy_err = _analyze_hy_ast(trees, tool_names)
    if hy_err is not None:
        return hy_err

    # Stage 3: Compile Hy AST to Python AST
    logger.debug("pipeline_compile")
    try:
        hy_input = (
            trees[0]
            if len(trees) == 1
            else Expression([Symbol("do"), *trees])
        )
        py_tree = hy_compile(hy_input, "__main__")
    except Exception as exc:
        return PipelineError(
            stage=STAGE_COMPILE,
            error_type=type(exc).__name__,
            message=str(exc),
            hy_source=source,
        )

    # Strip auto-injected 'import hy'
    py_tree.body = [
        n
        for n in py_tree.body
        if not isinstance(n, ast.Import)
        or n.names[0].name != "hy"
    ]

    # Stage 4: Python AST security analysis
    logger.debug("pipeline_ast_analysis")
    py_err = _analyze_python_ast(py_tree, tool_names)
    if py_err is not None:
        return py_err

    # Stage 5: Inject result capture
    py_tree = _inject_result_capture(py_tree)

    # Stage 6: Compile to bytecode
    try:
        bytecode = compile(py_tree, "<tgirl-pipeline>", "exec")
    except Exception as exc:
        return PipelineError(
            stage=STAGE_COMPILE,
            error_type=type(exc).__name__,
            message=str(exc),
            hy_source=source,
        )

    # Stage 7: Build sandbox
    sandbox = _build_sandbox(registry)

    # Apply per-tool timeouts
    for name in registry.names():
        tool_def = registry.get(name)
        if tool_def.timeout is not None:
            sandbox[name] = _wrap_with_timeout(
                sandbox[name], tool_def.timeout
            )

    # Stage 8: Execute with pipeline timeout
    logger.debug("pipeline_execute")

    def _execute() -> Any:
        try:
            exec(bytecode, sandbox)  # noqa: S102
        except Exception as exc:
            return PipelineError(
                stage=STAGE_EXECUTE,
                error_type=type(exc).__name__,
                message=str(exc),
                hy_source=source,
            )
        return sandbox["_tgirl_result_"]

    result = _run_with_timeout(_execute, cfg.pipeline_timeout)
    elapsed = (time.monotonic() - start_time) * 1000

    if isinstance(result, PipelineError):
        return result

    # Check for InsufficientResources
    if isinstance(result, InsufficientResources):
        return InsufficientResources(
            reason=result.reason, hy_source=source
        )

    # Check for runtime exceptions wrapped as PipelineError
    logger.info(
        "pipeline_complete",
        elapsed_ms=elapsed,
        source=source,
    )
    return PipelineResult(
        result=result,
        hy_source=source,
        execution_time_ms=elapsed,
    )
