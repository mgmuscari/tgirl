"""Dynamic CFG generation from registry state.

Converts RegistrySnapshot objects into context-free grammars in Lark EBNF
format that constrain LLM token output to only produce well-formed Hy
s-expressions invoking registered tools.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import re
from collections.abc import Callable, Mapping

import jinja2
import structlog
from pydantic import BaseModel, ConfigDict

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
    reachable_tokens: frozenset[int] | None = None


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


# --- Type-to-production converter ---


def _type_name_slug(type_repr: TypeRepr) -> str:
    """Generate a deterministic name slug from a TypeRepr."""
    if isinstance(type_repr, PrimitiveType):
        return type_repr.kind
    if isinstance(type_repr, ListType):
        return f"list_{_type_name_slug(type_repr.element)}"
    if isinstance(type_repr, DictType):
        k = _type_name_slug(type_repr.key)
        v = _type_name_slug(type_repr.value)
        return f"dict_{k}_{v}"
    if isinstance(type_repr, LiteralType):
        return f"lit_{'_'.join(str(x) for x in type_repr.values)}"
    if isinstance(type_repr, EnumType):
        return f"enum_{type_repr.name.lower()}"
    if isinstance(type_repr, OptionalType):
        return f"opt_{_type_name_slug(type_repr.inner)}"
    if isinstance(type_repr, UnionType):
        parts = "_".join(_type_name_slug(m) for m in type_repr.members)
        return f"union_{parts}"
    if isinstance(type_repr, ModelType):
        return f"model_{type_repr.name.lower()}"
    if isinstance(type_repr, AnnotatedType):
        return f"ann_{_type_name_slug(type_repr.base)}"
    if isinstance(type_repr, AnyType):
        return "any"
    msg = f"Unknown TypeRepr: {type_repr}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def _literal_value_to_grammar(val: str | int | float | bool) -> str:
    """Convert a literal value to a Lark grammar alternative."""
    if isinstance(val, bool):
        return f'"{"True" if val else "False"}"'
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"\\"{escaped}\\""'
    return f'"{val}"'


def _type_to_rule(
    type_repr: TypeRepr,
    rule_name: str,
    config: GrammarConfig,
) -> list[Production]:
    """Convert a TypeRepr to one or more grammar productions.

    Args:
        type_repr: The type to convert.
        rule_name: Name for the top-level production rule.
        config: Grammar generation configuration.

    Returns:
        List of Production objects (may include sub-rules for recursive types).
    """
    if isinstance(type_repr, PrimitiveType):
        rule_map = {
            "str": "ESCAPED_STRING",
            "int": "SIGNED_INT",
            "float": "SIGNED_FLOAT",
            "bool": '"True" | "False"',
            "none": '"nil"',
        }
        return [Production(name=rule_name, rule=rule_map[type_repr.kind])]

    if isinstance(type_repr, ListType):
        elem_name = f"{rule_name}_elem"
        elem_prods = _type_to_rule(type_repr.element, elem_name, config)
        rule = f'"[" ({elem_name} (SPACE {elem_name})*)? "]"'
        return [Production(name=rule_name, rule=rule)] + elem_prods

    if isinstance(type_repr, DictType):
        key_name = f"{rule_name}_key"
        val_name = f"{rule_name}_val"
        key_prods = _type_to_rule(type_repr.key, key_name, config)
        val_prods = _type_to_rule(type_repr.value, val_name, config)
        pair = f"{key_name} SPACE {val_name}"
        rule = f'"{{" ({pair} (SPACE {pair})*)? "}}"'
        return [Production(name=rule_name, rule=rule)] + key_prods + val_prods

    if isinstance(type_repr, LiteralType):
        parts = [_literal_value_to_grammar(v) for v in type_repr.values]
        return [Production(name=rule_name, rule=" | ".join(parts))]

    if isinstance(type_repr, EnumType):
        parts = []
        for ev in type_repr.values:
            escaped = ev.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"\\"{escaped}\\""')
        return [Production(name=rule_name, rule=" | ".join(parts))]

    if isinstance(type_repr, OptionalType):
        inner_name = f"{rule_name}_inner"
        inner_prods = _type_to_rule(type_repr.inner, inner_name, config)
        rule = f'{inner_name} | "nil"'
        return [Production(name=rule_name, rule=rule)] + inner_prods

    if isinstance(type_repr, UnionType):
        member_names = []
        all_prods: list[Production] = []
        for i, m in enumerate(type_repr.members):
            m_name = f"{rule_name}_m{i}"
            member_names.append(m_name)
            all_prods.extend(_type_to_rule(m, m_name, config))
        rule = " | ".join(member_names)
        return [Production(name=rule_name, rule=rule)] + all_prods

    if isinstance(type_repr, ModelType):
        field_prods: list[Production] = []
        required_parts: list[str] = []
        optional_parts: list[str] = []
        for fd in type_repr.fields:
            fv_name = f"{rule_name}_f_{fd.name}"
            fd_prods = _type_to_rule(fd.type_repr, fv_name, config)
            field_prods.extend(fd_prods)
            pair = f'"\\"{fd.name}\\"" SPACE {fv_name}'
            if fd.required:
                required_parts.append(pair)
            else:
                optional_parts.append(pair)
        all_parts = required_parts + optional_parts
        if all_parts:
            body = " SPACE ".join(all_parts)
            rule = f'"{{" {body} "}}"'
        else:
            rule = '"{"  "}"'
        return [Production(name=rule_name, rule=rule)] + field_prods

    if isinstance(type_repr, AnnotatedType):
        if not type_repr.constraints:
            return _type_to_rule(type_repr.base, rule_name, config)

        # Check for enumerable integer range
        if isinstance(type_repr.base, PrimitiveType) and type_repr.base.kind == "int":
            lo: int | None = None
            hi: int | None = None
            for c in type_repr.constraints:
                if c.kind == "ge":
                    lo = int(c.value)
                elif c.kind == "gt":
                    lo = int(c.value) + 1
                elif c.kind == "le":
                    hi = int(c.value)
                elif c.kind == "lt":
                    hi = int(c.value) - 1

            if lo is not None and hi is not None:
                range_size = hi - lo + 1
                if 0 < range_size <= config.enumeration_threshold:
                    parts = [f'"{i}"' for i in range(lo, hi + 1)]
                    return [
                        Production(
                            name=rule_name,
                            rule=" | ".join(parts),
                        )
                    ]

        # Fall back to base type
        return _type_to_rule(type_repr.base, rule_name, config)

    if isinstance(type_repr, AnyType):
        rule = 'ESCAPED_STRING | SIGNED_INT | SIGNED_FLOAT | "True" | "False" | "nil"'
        return [Production(name=rule_name, rule=rule)]

    msg = f"Unknown TypeRepr: {type_repr}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def _tool_to_rules(
    tool: ToolDefinition,
    config: GrammarConfig,
    type_grammars: Mapping[str, str] | None = None,
) -> list[Production]:
    """Convert a ToolDefinition to grammar productions.

    Uses positional, trailing-optional chain convention:
    required params first in fixed order, then optional params
    as a nested optional chain.

    When a parameter has a semantic type tag with a registered grammar
    rule, the tagged rule is used instead of the default type rule.

    Args:
        tool: The tool definition.
        config: Grammar generation configuration.
        type_grammars: Mapping of semantic type tags to Lark EBNF rules.

    Returns:
        List of Production objects for this tool.
    """
    prods: list[Production] = []
    tag_map = dict(tool.param_tags)
    tg = type_grammars or {}

    # Split required and optional parameters
    required = [p for p in tool.parameters if not p.has_default]
    optional = [p for p in tool.parameters if p.has_default]

    # Generate type productions for each parameter
    param_type_names: list[str] = []
    for param in tool.parameters:
        tag = tag_map.get(param.name)
        if tag and tag in tg:
            # Use the semantic type's shared rule name
            rule_name = f"stype_{tag.lower()}"
            param_type_names.append(rule_name)
            # The actual rule definition is emitted once (deduped later)
            prods.append(Production(name=rule_name, rule=tg[tag]))
        else:
            type_name = f"param_{tool.name}_{param.name}".lower()
            param_type_names.append(type_name)
            prods.extend(_type_to_rule(param.type_repr, type_name, config))

    if not tool.parameters:
        # No parameters: (tool_name)
        rule = f'"(" "{tool.name}" ")"'
        prods.insert(0, Production(name=f"call_{tool.name}".lower(), rule=rule))
        return prods

    # Build the args portion with trailing optional chain
    # Required params are positional, optional params nest
    req_count = len(required)
    opt_count = len(optional)

    # Build from innermost optional outward
    if opt_count > 0:
        # Start with the last optional param
        last_opt_idx = req_count + opt_count - 1
        chain = f"SPACE {param_type_names[last_opt_idx]}"
        # Wrap remaining optionals from inside out
        for i in range(last_opt_idx - 1, req_count - 1, -1):
            chain = f"SPACE {param_type_names[i]} ({chain})?"
    else:
        chain = None

    # Build the full args
    parts = [param_type_names[i] for i in range(req_count)]

    if chain is not None:
        args = (
            " SPACE ".join(parts) + f" ({chain})?" if req_count > 0 else f"({chain})?"
        )
    else:
        args = " SPACE ".join(parts)

    if req_count > 0:
        rule = f'"(" "{tool.name}" SPACE {args} ")"'
    else:
        # All-optional: space is part of the optional group
        rule = f'"(" "{tool.name}" {args} ")"'
    prods.insert(0, Production(name=f"call_{tool.name}".lower(), rule=rule))
    return prods


def _load_templates() -> jinja2.Environment:
    """Load Jinja2 templates from the templates package."""
    templates_path = importlib.resources.files("tgirl.templates")
    loader = jinja2.FileSystemLoader(str(templates_path))
    return jinja2.Environment(loader=loader, keep_trailing_newline=True)


def _render_grammar(
    snapshot: RegistrySnapshot,
    config: GrammarConfig,
) -> str:
    """Render the complete grammar from a snapshot.

    Args:
        snapshot: Registry snapshot to render grammar for.
        config: Grammar generation configuration.

    Returns:
        Complete grammar text in Lark EBNF format.
    """
    env = _load_templates()

    # Collect all productions
    tool_prods: list[Production] = []
    type_prods: list[Production] = []
    tool_call_names: list[str] = []

    tg = dict(snapshot.type_grammars)

    for tool in snapshot.tools:
        rules = _tool_to_rules(tool, config, type_grammars=tg)
        # First production is the call_<name> rule
        tool_call_names.append(rules[0].name)
        tool_prods.append(rules[0])
        # Remaining are parameter type productions
        for p in rules[1:]:
            type_prods.append(p)

    # Deduplicate type productions by name (keep first occurrence)
    seen: set[str] = set()
    deduped_type_prods: list[Production] = []
    for p in type_prods:
        if p.name not in seen:
            seen.add(p.name)
            deduped_type_prods.append(p)

    tool_alternatives = " | ".join(tool_call_names) if tool_call_names else ""

    template = env.get_template("base.cfg.j2")
    return template.render(
        tool_alternatives=tool_alternatives,
        tool_productions=tool_prods,
        type_productions=deduped_type_prods,
    )


def generate_routing_grammar(
    snapshot: RegistrySnapshot,
    top_k: int = 1,
) -> str:
    """Generate a minimal grammar that accepts tool names.

    When top_k=1 (default), produces a single-choice grammar:
        start: tool_choice
        tool_choice: "alpha" | "beta" | "gamma"

    When top_k > 1, produces a space-separated list grammar:
        start: tool_list
        tool_list: tool_choice (" " tool_choice){0,top_k-1}
        tool_choice: "alpha" | "beta" | "gamma"

    Args:
        snapshot: Registry snapshot with available tools.
        top_k: Maximum number of tools in the output list.

    Returns Lark EBNF text (not a GrammarOutput -- no productions or hash needed).
    """
    if not snapshot.tools:
        msg = "Cannot generate routing grammar for empty snapshot"
        raise ValueError(msg)
    alternatives = " | ".join(f'"{tool.name}"' for tool in snapshot.tools)
    if top_k <= 1:
        return f"start: tool_choice\ntool_choice: {alternatives}\n"
    # Build trailing optional chain for bounded repetition.
    # Uses the same nested-optional pattern as _tool_to_rules for
    # optional parameters — compatible with both Lark LALR(1) and llguidance.
    # e.g., top_k=3: tool_choice (" " tool_choice (" " tool_choice)?)?
    chain = '" " tool_choice'
    for _ in range(top_k - 2):
        chain = f'" " tool_choice ({chain})?'
    return (
        f"start: tool_list\n"
        f"tool_list: tool_choice ({chain})?\n"
        f"tool_choice: {alternatives}\n"
    )


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
    """
    cfg = config or GrammarConfig()

    # Collect all productions
    tg = dict(snapshot.type_grammars)
    all_prods: list[Production] = []
    for tool in snapshot.tools:
        all_prods.extend(_tool_to_rules(tool, cfg, type_grammars=tg))

    # Also collect type productions from return types
    for tool in snapshot.tools:
        ret_name = f"ret_{tool.name}".lower()
        all_prods.extend(_type_to_rule(tool.return_type, ret_name, cfg))

    # Deduplicate by name (keep first occurrence)
    seen: set[str] = set()
    deduped: list[Production] = []
    for p in all_prods:
        if p.name not in seen:
            seen.add(p.name)
            deduped.append(p)

    # Render grammar text
    text = _render_grammar(snapshot, cfg)

    # Compute snapshot hash (exclude timestamp for determinism)
    hash_data = snapshot.model_dump_json(exclude={"timestamp"})
    snapshot_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    return GrammarOutput(
        text=text,
        productions=tuple(deduped),
        snapshot_hash=snapshot_hash,
        tool_quotas=dict(snapshot.quotas),
        cost_remaining=snapshot.cost_remaining,
    )


def compute_reachable_set(
    grammar_text: str,
    tokenizer_decode: Callable[[list[int]], str],
    vocab_size: int,
) -> frozenset[int]:
    """Compute the set of token IDs reachable from any grammar terminal.

    Extracts string literals and regex terminals from the grammar text,
    then scans the vocabulary to find all tokens that could match any
    terminal. Returns a frozenset of reachable token IDs.

    The reachable set restricts OT to operate on a much smaller problem,
    raising engagement from ~20% to >80% of constrained generation tokens.
    """
    if not grammar_text or vocab_size <= 0:
        return frozenset()

    # 1. Extract characters/strings that can appear in the grammar
    reachable_chars: set[str] = set()
    reachable_strings: set[str] = set()

    # Extract string literals from Lark EBNF: "..." patterns
    for match in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', grammar_text):
        literal = match.group(1)
        reachable_strings.add(literal)
        reachable_chars.update(literal)

    # Extract regex terminal patterns and derive their character sets
    regex_patterns: list[re.Pattern] = []
    for match in re.finditer(
        r'(\w+)\s*:\s*/([^/]+)/', grammar_text
    ):
        pattern_str = match.group(2)
        try:
            regex_patterns.append(re.compile(pattern_str))
        except re.error:
            continue
        # Extract character classes from regex
        for cm in re.finditer(r'\[([^\]]+)\]', pattern_str):
            char_class = cm.group(1)
            # Expand simple ranges like 0-9, a-z, A-Z
            i = 0
            while i < len(char_class):
                if (
                    i + 2 < len(char_class)
                    and char_class[i + 1] == '-'
                ):
                    start = ord(char_class[i])
                    end = ord(char_class[i + 2])
                    for c in range(start, end + 1):
                        reachable_chars.add(chr(c))
                    i += 3
                elif char_class[i] == '\\':
                    if i + 1 < len(char_class):
                        reachable_chars.add(char_class[i + 1])
                    i += 2
                else:
                    reachable_chars.add(char_class[i])
                    i += 1
        # Standalone chars in regex (outside classes)
        for ch in re.sub(r'\[.*?\]|\{.*?\}|[().|*+?^$\\]', '', pattern_str):
            if ch.isalnum() or ch in '+-_.':
                reachable_chars.add(ch)

    # Handle %import common.ESCAPED_STRING — matches printable ASCII
    if 'ESCAPED_STRING' in grammar_text:
        reachable_chars.add('"')
        reachable_chars.add('\\')
        for c in range(32, 127):
            reachable_chars.add(chr(c))

    # 2. Scan vocabulary
    reachable: set[int] = set()
    for tid in range(vocab_size):
        try:
            text = tokenizer_decode([tid])
        except Exception:
            continue
        if not text:
            continue

        # Check exact string match
        if text in reachable_strings:
            reachable.add(tid)
            continue

        # Check if all chars in the token are reachable chars
        if all(c in reachable_chars for c in text):
            reachable.add(tid)
            continue

        # Check regex match for single-token values
        for pat in regex_patterns:
            if pat.fullmatch(text) or pat.fullmatch(text.strip()):
                reachable.add(tid)
                break

    return frozenset(reachable)


def diff(a: GrammarOutput, b: GrammarOutput) -> GrammarDiff:
    """Compute the diff between two grammars.

    Args:
        a: First grammar.
        b: Second grammar.

    Returns:
        Diff showing added, removed, and changed productions.
    """
    a_map = {p.name: p for p in a.productions}
    b_map = {p.name: p for p in b.productions}

    a_names = set(a_map)
    b_names = set(b_map)

    added = tuple(b_map[n] for n in sorted(b_names - a_names))
    removed = tuple(a_map[n] for n in sorted(a_names - b_names))
    changed = tuple(
        (a_map[n], b_map[n])
        for n in sorted(a_names & b_names)
        if a_map[n].rule != b_map[n].rule
    )

    return GrammarDiff(added=added, removed=removed, changed=changed)
