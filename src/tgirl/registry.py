"""Tool registration, type extraction, and snapshot generation."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import structlog

from tgirl._type_extract import extract_parameters
from tgirl.types import (
    AnyType,
    DictType,
    FieldDef,
    ListType,
    LiteralType,
    ModelType,
    ParameterDef,
    PrimitiveType,
    RegistrySnapshot,
    ToolDefinition,
    TypeRepr,
)

# Audit finding #6: tool names appear as quoted Lark string terminals in
# generated grammars. A name containing a quote character splits one
# terminal into multiple, opening grammar-drift attacks. Validate at
# registration time with a charset that's safe to drop into a Lark
# string-terminal slot AND legible as an s-expression head identifier.
#
# Pattern: leading letter (matches Python identifier shape and avoids leading
# digits that would be ambiguous with grammar rule numerics), followed by
# letters / digits / underscore / dot / hyphen. Plugins commonly use dotted
# (``<plugin>.<tool>``) and hyphenated names; both are admitted.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")


def _validate_tool_name(name: str) -> None:
    """Reject tool names that aren't safe to emit as Lark string terminals.

    Single source of truth for the tool-name charset; called from both
    ``ToolRegistry.tool()`` and ``ToolRegistry.register_from_schema()``.
    """
    if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
        msg = (
            f"Tool name {name!r} is invalid; must match "
            f"{_TOOL_NAME_RE.pattern!r} (start with a letter; allowed "
            "characters: letters, digits, underscore, dot, hyphen). "
            "This rule prevents Lark grammar terminal-injection."
        )
        raise ValueError(msg)

logger = structlog.get_logger()

_JSON_SCHEMA_TYPE_MAP: dict[str, TypeRepr] = {
    "string": PrimitiveType(kind="str"),
    "integer": PrimitiveType(kind="int"),
    "number": PrimitiveType(kind="float"),
    "float": PrimitiveType(kind="float"),
    "boolean": PrimitiveType(kind="bool"),
    "any": AnyType(),
}


def _schema_type_to_repr(prop_schema: dict[str, Any]) -> TypeRepr:
    """Convert a JSON schema property definition to a TypeRepr."""
    # Check for enum first — overrides type
    if "enum" in prop_schema:
        return LiteralType(values=tuple(prop_schema["enum"]))

    schema_type = prop_schema.get("type", "any")

    if schema_type in _JSON_SCHEMA_TYPE_MAP:
        return _JSON_SCHEMA_TYPE_MAP[schema_type]

    if schema_type == "array":
        items = prop_schema.get("items")
        if items:
            return ListType(element=_schema_type_to_repr(items))
        return ListType(element=AnyType())

    if schema_type == "object":
        properties = prop_schema.get("properties")
        if properties:
            required_names = set(prop_schema.get("required", []))
            fields = tuple(
                FieldDef(
                    name=fname,
                    type_repr=_schema_type_to_repr(fschema),
                    required=fname in required_names,
                    default=fschema.get("default"),
                )
                for fname, fschema in properties.items()
            )
            name_hash = "_".join(sorted(properties.keys()))[:32]
            return ModelType(name=f"Object_{name_hash}", fields=fields)
        return DictType(
            key=PrimitiveType(kind="str"), value=AnyType()
        )

    if schema_type == "dict":
        return DictType(
            key=PrimitiveType(kind="str"), value=AnyType()
        )

    if schema_type == "null":
        return PrimitiveType(kind="none")

    if schema_type == "tuple":
        return ListType(element=AnyType())

    return AnyType()


class ToolRegistry:
    """Mutable registry for tool definitions.

    Tools are registered via the `tool()` decorator at startup time.
    Immutable snapshots are produced via `snapshot()` for each
    generation request.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._callables: dict[str, Callable[..., Any]] = {}
        self._type_grammars: dict[str, str] = {}
        # Parallel dict: tool_name → source plugin name (or "inline" / "at_tool_kwarg").
        # PRP Task 10 §"Approach" — enables plugin-level dedup and Task 11
        # `/telemetry` `source` field.
        self._sources: dict[str, str] = {}
        # Parallel dict: tool_name → CapabilityGrant. Audit finding #1 / Phase B.
        # When a tool is registered with a grant (i.e. via plugin loading), the
        # grant is recorded here so ``get_callable`` can scope ``guard_scope``
        # around invocation. Tools registered via the ``@tool()`` decorator
        # without plugin context (e.g. host-app inline registration) have no
        # entry — ``get_callable`` returns the raw callable in that case.
        from tgirl.plugins.types import (  # local import: avoids cycle
            CapabilityGrant,
        )

        self._grants: dict[str, CapabilityGrant] = {}
        # Forward-declare the ImportError-pinned cycle so mypy sees the type.
        _ = CapabilityGrant

    def tool(
        self,
        *,
        quota: int | None = None,
        cost: float = 0.0,
        cost_budget: float | None = None,
        scope: str | None = None,
        timeout: float | None = None,
        cacheable: bool = False,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers a function as a tool.

        The function is returned unmodified — this is a metadata-only
        decorator. All type information is extracted from annotations.

        Args:
            quota: Maximum number of calls per pipeline, or None.
            cost: Cost per invocation.
            cost_budget: Per-tool cost budget, or None.
            scope: Authorization scope, or None for unrestricted.
            timeout: Execution timeout in seconds, or None.
            cacheable: Whether results can be cached.
            description: Human-readable description.

        Returns:
            The original function, unmodified.

        Raises:
            ValueError: If a tool with the same name is already registered.
            TypeError: If any parameter lacks a type annotation or
                the function lacks a return type annotation.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            name = func.__name__

            _validate_tool_name(name)
            if name in self._tools:
                msg = f"Tool '{name}' is already registered"
                raise ValueError(msg)

            params, return_type = extract_parameters(
                func, include_return=True
            )

            tool_def = ToolDefinition(
                name=name,
                parameters=params,
                return_type=return_type,
                quota=quota,
                cost=cost,
                cost_budget=cost_budget,
                scope=scope,
                timeout=timeout,
                cacheable=cacheable,
                description=description or func.__doc__ or "",
            )

            self._tools[name] = tool_def
            self._callables[name] = func

            logger.debug(
                "tool_registered",
                name=name,
                params=len(params),
                quota=quota,
                scope=scope,
            )

            return func

        return decorator

    def register_from_schema(
        self,
        name: str,
        parameters: dict[str, Any],
        description: str = "",
        *,
        return_type: TypeRepr | None = None,
        quota: int | None = None,
        cost: float = 0.0,
        scope: str | None = None,
        timeout: float | None = None,
        callable_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Register a tool from a JSON schema definition.

        Args:
            name: Tool name (pre-sanitized by caller).
            parameters: JSON schema object with 'properties' and 'required'.
            description: Human-readable description.
            return_type: Return type. Defaults to AnyType.
            quota: Maximum number of calls per pipeline, or None.
            cost: Cost per invocation.
            scope: Authorization scope, or None for unrestricted.
            timeout: Execution timeout in seconds, or None.
            callable_fn: Execution callable. Defaults to no-op.

        Raises:
            ValueError: If a tool with the same name is already registered,
                or if the name violates the registration-charset rule
                (audit finding #6).
        """
        _validate_tool_name(name)
        if name in self._tools:
            msg = f"Tool '{name}' is already registered"
            raise ValueError(msg)

        properties = parameters.get("properties", {})
        required_names = set(parameters.get("required", []))

        # Build required params first, then optional (preserves ordering)
        required_params: list[ParameterDef] = []
        optional_params: list[ParameterDef] = []

        for param_name, prop_schema in properties.items():
            type_repr = _schema_type_to_repr(prop_schema)
            is_required = param_name in required_names
            param = ParameterDef(
                name=param_name,
                type_repr=type_repr,
                has_default=not is_required,
                default=None,
                description=prop_schema.get("description", ""),
            )
            if is_required:
                required_params.append(param)
            else:
                optional_params.append(param)

        params = tuple(required_params + optional_params)

        tool_def = ToolDefinition(
            name=name,
            parameters=params,
            return_type=return_type or AnyType(),
            quota=quota,
            cost=cost,
            scope=scope,
            timeout=timeout,
            description=description,
        )

        self._tools[name] = tool_def
        self._callables[name] = (
            callable_fn if callable_fn is not None else lambda **kwargs: None
        )

        logger.debug(
            "tool_registered_from_schema",
            name=name,
            params=len(params),
            quota=quota,
            scope=scope,
        )

    def snapshot(
        self,
        *,
        scopes: set[str] | None = None,
        restrict_to: list[str] | None = None,
        cost_budget: float | None = None,
    ) -> RegistrySnapshot:
        """Produce an immutable snapshot of registry state.

        Args:
            scopes: Authorized scopes. Tools with scope=None are always
                included. If None, all tools are included.
            restrict_to: If provided, only include named tools.
                Unknown names are silently ignored.
            cost_budget: Cost budget for this pipeline execution.

        Returns:
            A frozen RegistrySnapshot.
        """
        tools: list[ToolDefinition] = []
        quotas: dict[str, int] = {}

        for name in sorted(self._tools):
            td = self._tools[name]

            # Scope filtering
            if (
                scopes is not None
                and td.scope is not None
                and td.scope not in scopes
            ):
                continue

            # Restrict-to filtering
            if restrict_to is not None and name not in restrict_to:
                continue

            tools.append(td)
            if td.quota is not None:
                quotas[name] = td.quota

        # Collect type grammars referenced by included tools
        referenced_tags = set()
        for td in tools:
            for _param_name, tag in td.param_tags:
                referenced_tags.add(tag)

        type_grammars = tuple(
            (tag, rule) for tag, rule in sorted(self._type_grammars.items())
            if tag in referenced_tags
        )

        return RegistrySnapshot(
            tools=tuple(tools),
            quotas=quotas,
            cost_remaining=cost_budget,
            scopes=frozenset(scopes) if scopes is not None else frozenset(),
            timestamp=time.time(),
            type_grammars=type_grammars,
        )

    def register_type(self, tag: str, grammar_rule: str) -> None:
        """Register a semantic type with a grammar rule.

        Semantic types create distinct non-terminals in the generated
        grammar. When a parameter is tagged with a registered type,
        the grammar uses the type's rule instead of the default
        terminal (e.g., ESCAPED_STRING).

        Args:
            tag: The semantic type name (e.g., "JsonObject", "FieldName").
            grammar_rule: Lark EBNF rule body for this type.

        Raises:
            ValueError: If the tag is already registered.
        """
        if tag in self._type_grammars:
            msg = f"Type '{tag}' is already registered"
            raise ValueError(msg)
        self._type_grammars[tag] = grammar_rule

    def enrich(
        self,
        name: str,
        *,
        param_tags: dict[str, str] | None = None,
        examples: list[str] | None = None,
    ) -> None:
        """Add semantic metadata to a registered tool.

        Enrichment is additive — it adds type tags and usage examples
        without changing the tool's type signature or behavior.

        Args:
            name: Name of the registered tool.
            param_tags: Mapping of parameter name to semantic type tag.
            examples: Usage examples as s-expression strings.

        Raises:
            KeyError: If the tool is not registered.
            ValueError: If a param_tags key doesn't match any parameter.
        """
        if name not in self._tools:
            raise KeyError(name)

        tool = self._tools[name]
        updates: dict[str, object] = {}

        if param_tags:
            param_names = {p.name for p in tool.parameters}
            unknown = set(param_tags) - param_names
            if unknown:
                bad = next(iter(unknown))
                msg = f"{bad}"
                raise ValueError(msg)

            merged = dict(tool.param_tags)
            merged.update(param_tags)
            updates["param_tags"] = tuple(merged.items())

        if examples:
            updates["examples"] = tuple(examples)

        if updates:
            self._tools[name] = tool.model_copy(update=updates)

    def get(self, name: str) -> ToolDefinition:
        """Get a tool definition by name.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def get_callable(self, name: str) -> Callable[..., Any]:
        """Get the registered callable for a tool.

        Audit finding #1 / Phase B: if the tool was registered with a
        ``CapabilityGrant`` (i.e. via plugin loading), the returned callable
        wraps the underlying invocation in a ``guard_scope(grant)`` block so
        that Gate 2 (meta_path finder) and Gate 3 (CapabilityScopedModule)
        observe the grant during execution.

        Tools registered via the bare ``@tool()`` decorator without plugin
        context have no recorded grant; ``get_callable`` returns the raw
        callable for them — preserving the host-app convention of
        unconstrained inline tools.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._callables:
            raise KeyError(name)
        raw = self._callables[name]
        grant = self._grants.get(name)
        if grant is None:
            return raw

        # Phase B wrapper: enter guard_scope around every invocation. The
        # wrapper preserves the original callable's signature transparently;
        # callers receive the same return value, with the same exception
        # propagation semantics.
        from tgirl.plugins.guard import guard_scope  # local: avoids cycle

        def _grant_scoped(*args: Any, **kwargs: Any) -> Any:
            with guard_scope(grant):
                return raw(*args, **kwargs)

        # Surface the wrapped callable's identity so introspection still
        # shows the underlying name. Defensive — registry callers that
        # rely on ``__name__`` won't be surprised.
        _grant_scoped.__name__ = getattr(raw, "__name__", name)
        _grant_scoped.__qualname__ = getattr(raw, "__qualname__", name)
        return _grant_scoped

    def sanitized_rule_names(self) -> dict[str, str]:
        """Return ``{tool_name: sanitized_rule_slug}`` for every registered tool.

        Shared source of truth for grammar and instructions modules. Used to
        detect sanitized-name collisions symmetrically (PRP Task 10, Y#4).
        """
        from tgirl.grammar import _sanitize_rule_name
        return {name: _sanitize_rule_name(name) for name in self._tools}

    def names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
