"""Tool registration, type extraction, and snapshot generation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog

from tgirl._type_extract import extract_parameters
from tgirl.types import (
    AnyType,
    DictType,
    ListType,
    LiteralType,
    ParameterDef,
    PrimitiveType,
    RegistrySnapshot,
    ToolDefinition,
    TypeRepr,
)

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

    if schema_type == "dict":
        return DictType(
            key=PrimitiveType(kind="str"), value=AnyType()
        )

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
    ) -> None:
        """Register a tool from a JSON schema definition.

        Args:
            name: Tool name (pre-sanitized by caller).
            parameters: JSON schema object with 'properties' and 'required'.
            description: Human-readable description.
            return_type: Return type. Defaults to AnyType.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
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
                default=None if not is_required else None,
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
            description=description,
        )

        self._tools[name] = tool_def
        self._callables[name] = lambda **kwargs: None

        logger.debug(
            "tool_registered_from_schema",
            name=name,
            params=len(params),
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
            for param_name, tag in td.param_tags:
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
        """Get the original callable for a registered tool.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._callables:
            raise KeyError(name)
        return self._callables[name]

    def names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
