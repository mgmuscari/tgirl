"""Instruction generator for tool-calling system prompts.

Auto-generates structured system prompts from RegistrySnapshot,
using parameter names, types, descriptions, and optional enrichment
tags to help models distinguish between tools with similar type
signatures.
"""

from __future__ import annotations

import structlog

from tgirl.types import (
    AnnotatedType,
    AnyType,
    DictType,
    EnumType,
    ListType,
    LiteralType,
    ModelType,
    OptionalType,
    PrimitiveType,
    RegistrySnapshot,
    ToolDefinition,
    TypeRepr,
    UnionType,
)

logger = structlog.get_logger()


def _type_label(type_repr: TypeRepr) -> str:
    """Human-readable label for a type."""
    if isinstance(type_repr, PrimitiveType):
        return type_repr.kind
    if isinstance(type_repr, ListType):
        return f"list[{_type_label(type_repr.element)}]"
    if isinstance(type_repr, DictType):
        return f"dict[{_type_label(type_repr.key)}, {_type_label(type_repr.value)}]"
    if isinstance(type_repr, LiteralType):
        vals = " | ".join(repr(v) for v in type_repr.values)
        return f"Literal[{vals}]"
    if isinstance(type_repr, EnumType):
        return type_repr.name
    if isinstance(type_repr, OptionalType):
        return f"Optional[{_type_label(type_repr.inner)}]"
    if isinstance(type_repr, UnionType):
        parts = " | ".join(_type_label(m) for m in type_repr.members)
        return f"Union[{parts}]"
    if isinstance(type_repr, ModelType):
        return type_repr.name
    if isinstance(type_repr, AnnotatedType):
        return _type_label(type_repr.base)
    if isinstance(type_repr, AnyType):
        return "any"
    return "unknown"


def _param_label(name: str, type_repr: TypeRepr, tag: str | None = None) -> str:
    """Format a parameter as <name:Type> or <name:Tag> if enriched."""
    type_name = tag if tag else _type_label(type_repr)
    return f"<{name}:{type_name}>"


def generate_tool_doc(tool: ToolDefinition) -> str:
    """Generate documentation for a single tool.

    Uses parameter names and optional enrichment tags to produce
    distinctive signatures that help models distinguish tools
    with similar underlying types.

    Args:
        tool: The tool definition (with optional enrichment metadata).

    Returns:
        Formatted tool documentation string.
    """
    tags: dict[str, str] = dict(tool.param_tags)

    # Build s-expression signature
    param_parts = []
    for p in tool.parameters:
        tag = tags.get(p.name)
        param_parts.append(_param_label(p.name, p.type_repr, tag))

    params_str = " ".join(param_parts)
    if params_str:
        signature = f"({tool.name} {params_str})"
    else:
        signature = f"({tool.name})"

    ret_label = _type_label(tool.return_type)

    lines = [
        f"  {signature} -> {ret_label}",
        f"    {tool.description}",
    ]

    for p in tool.parameters:
        if p.description:
            default_note = " (optional)" if p.has_default else ""
            lines.append(f"    - {p.name}: {p.description}{default_note}")

    for example in tool.examples:
        lines.append(f"    Example: {example}")

    return "\n".join(lines)


def generate_system_prompt(
    snapshot: RegistrySnapshot,
    tool_open: str | None = None,
    tool_close: str | None = None,
) -> str:
    """Generate a complete system prompt from a registry snapshot.

    Produces structured instructions including:
    - S-expression syntax explanation
    - Tool signatures with parameter names and types
    - Tool descriptions
    - Delimiter protocol (when tool_open/tool_close are provided)

    Args:
        snapshot: Immutable registry snapshot.
        tool_open: Opening delimiter for tool calls (e.g. ``<tool>``).
            When provided, the prompt instructs the model to wrap
            tool calls in these delimiters.
        tool_close: Closing delimiter for tool calls (e.g. ``</tool>``).

    Returns:
        Complete system prompt string.
    """
    sections = [
        "You are a tool-calling assistant. You call tools using "
        "s-expressions wrapped in delimiters."
        if tool_open
        else "You call tools using s-expressions. "
        "Reply with ONLY one s-expression, nothing else.",
        "",
        "## Syntax",
        "  (tool_name arg1 arg2 ...)",
        "  Strings are double-quoted. Integers are bare numbers.",
        "",
    ]

    if tool_open and tool_close:
        sections.extend([
            "## Tool Call Format",
            f"  Wrap every tool call in {tool_open}...{tool_close} delimiters:",
            f"  {tool_open}(tool_name arg1 arg2){tool_close}",
            "",
            "  When you want to call a tool, output the delimiter, then the "
            "s-expression, then the closing delimiter. You may include "
            "natural language before the tool call.",
            "",
        ])

    sections.extend(["## Available Tools", ""])

    for tool in snapshot.tools:
        sections.append(generate_tool_doc(tool))
        sections.append("")

    return "\n".join(sections).rstrip()


def generate_routing_prompt(snapshot: RegistrySnapshot) -> str:
    """Generate a routing prompt for tool selection.

    Composition-aware: instructs the model to list tools needed
    to fulfill the request, most relevant first. If only one tool
    is needed, list just that one.
    """
    lines = [
        "You are a tool routing assistant. "
        "Given a user request, list the tools needed to fulfill it, "
        "most relevant first.",
        "If multiple tools could be composed together to answer "
        "the request, list all of them.",
        "If only one tool is needed, list just that one.",
        "",
        "Available tools:",
    ]
    for tool in snapshot.tools:
        params = ", ".join(p.name for p in tool.parameters)
        lines.append(f"- {tool.name}({params}): {tool.description}")
    lines.append("")
    lines.append("Reply with ONLY tool names separated by spaces, nothing else.")
    return "\n".join(lines)
