"""Tool registration, type extraction, and snapshot generation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog

from tgirl._type_extract import extract_parameters
from tgirl.types import (
    RegistrySnapshot,
    ToolDefinition,
)

logger = structlog.get_logger()


class ToolRegistry:
    """Mutable registry for tool definitions.

    Tools are registered via the `tool()` decorator at startup time.
    Immutable snapshots are produced via `snapshot()` for each
    generation request.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

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

        return RegistrySnapshot(
            tools=tuple(tools),
            quotas=quotas,
            cost_remaining=cost_budget,
            scopes=frozenset(scopes) if scopes is not None else frozenset(),
            timestamp=time.time(),
        )

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
