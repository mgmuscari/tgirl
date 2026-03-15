"""BFCL benchmark adapter — schema loading and output translation."""

from __future__ import annotations

import importlib.resources
import json
from typing import Any

import hy.models
import structlog

from tgirl.registry import ToolRegistry

logger = structlog.get_logger()


def load_test_data(category: str) -> list[dict[str, Any]]:
    """Load BFCL JSONL test data for a category.

    Args:
        category: BFCL category name (e.g., 'simple_python').

    Returns:
        List of test entry dicts with 'id', 'question', 'function' keys.
    """
    data_dir = importlib.resources.files("bfcl_eval") / "data"
    path = data_dir / f"BFCL_v4_{category}.json"
    text = path.read_text()  # type: ignore[union-attr]
    return [json.loads(line) for line in text.strip().split("\n")]


def load_ground_truth(category: str) -> list[dict[str, Any]]:
    """Load BFCL ground truth entries for a category.

    Args:
        category: BFCL category name (e.g., 'simple_python').

    Returns:
        List of ground truth dicts with 'id' and 'ground_truth' keys.
    """
    data_dir = importlib.resources.files("bfcl_eval") / "data"
    path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
    text = path.read_text()  # type: ignore[union-attr]
    return [json.loads(line) for line in text.strip().split("\n")]


def register_bfcl_tools(
    registry: ToolRegistry,
    functions: list[dict[str, Any]],
) -> dict[str, str]:
    """Register BFCL function definitions into a ToolRegistry.

    Sanitizes dotted names (e.g., 'spotify.play' -> 'spotify_play')
    and returns a mapping from sanitized to original names.

    Args:
        registry: The ToolRegistry to register tools into.
        functions: List of BFCL function definition dicts.

    Returns:
        Dict mapping sanitized_name -> original_name.
    """
    name_map: dict[str, str] = {}
    for func_def in functions:
        original_name = func_def["name"]
        sanitized_name = original_name.replace(".", "_")
        name_map[sanitized_name] = original_name

        registry.register_from_schema(
            name=sanitized_name,
            parameters=func_def.get("parameters", {}),
            description=func_def.get("description", ""),
        )

    return name_map


def _hy_node_to_python(node: Any) -> Any:
    """Convert a Hy AST node to a Python value."""
    if isinstance(node, hy.models.Integer):
        return int(node)
    if isinstance(node, hy.models.Float):
        return float(node)
    if isinstance(node, hy.models.String):
        return str(node)
    if isinstance(node, hy.models.List):
        return [_hy_node_to_python(item) for item in node]
    if isinstance(node, hy.models.Dict):
        keys = list(node)[::2]
        vals = list(node)[1::2]
        return {
            _hy_node_to_python(k): _hy_node_to_python(v)
            for k, v in zip(keys, vals, strict=True)
        }
    if isinstance(node, hy.models.Symbol):
        sym = str(node)
        if sym in ("True", "true"):
            return True
        if sym in ("False", "false"):
            return False
        if sym in ("None", "nil"):
            return None
        return sym
    return node


def _format_python_value(val: Any) -> str:
    """Format a Python value as it would appear in Python source."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, str):
        # Use repr() for proper escaping of control chars, then
        # normalize to double quotes for BFCL compatibility
        r = repr(val)
        if r.startswith("'") and r.endswith("'"):
            inner = r[1:-1]
            # repr with single quotes doesn't escape double quotes,
            # but does escape single quotes — swap
            inner = inner.replace("\\'", "'").replace('"', '\\"')
            return f'"{inner}"'
        return r
    if isinstance(val, list):
        items = ", ".join(_format_python_value(v) for v in val)
        return f"[{items}]"
    if isinstance(val, dict):
        items = ", ".join(
            f"{_format_python_value(k)}: {_format_python_value(v)}"
            for k, v in val.items()
        )
        return "{" + items + "}"
    return repr(val)


def sexpr_to_bfcl(
    hy_source: str,
    registry: ToolRegistry,
    name_map: dict[str, str],
) -> str:
    """Convert tgirl s-expression output to BFCL Python-style format.

    Uses Hy AST parser for structural correctness.

    Args:
        hy_source: Hy s-expression string (one or more expressions).
        registry: ToolRegistry with the registered tools.
        name_map: Mapping from sanitized names to original dotted names.

    Returns:
        BFCL format string like '[func(param1=val1, param2=val2)]'.
    """
    expressions = list(hy.read_many(hy_source))
    call_strs: list[str] = []

    for expr in expressions:
        items = list(expr)
        func_symbol = str(items[0])
        args = [_hy_node_to_python(item) for item in items[1:]]

        # Look up parameter names from registry
        tool_def = registry.get(func_symbol)
        params = tool_def.parameters

        # Map positional args to named params
        named_args: list[str] = []
        for i, val in enumerate(args):
            if i < len(params):
                param_name = params[i].name
                named_args.append(
                    f"{param_name}={_format_python_value(val)}"
                )

        # Map sanitized name back to original
        original_name = name_map.get(func_symbol, func_symbol)
        call_str = f"{original_name}({', '.join(named_args)})"
        call_strs.append(call_str)

    return "[" + ", ".join(call_strs) + "]"


def sexpr_to_bfcl_dict(
    hy_source: str,
    registry: ToolRegistry,
    name_map: dict[str, str],
) -> list[dict[str, dict[str, Any]]]:
    """Convert tgirl s-expression output to BFCL checker format.

    Returns a list of dicts like:
        [{"func_name": {"param1": val1, "param2": val2}}]

    This is the format expected by bfcl_eval's ast_checker.
    """
    expressions = list(hy.read_many(hy_source))
    calls: list[dict[str, dict[str, Any]]] = []

    for expr in expressions:
        items = list(expr)
        func_symbol = str(items[0])
        args = [_hy_node_to_python(item) for item in items[1:]]

        tool_def = registry.get(func_symbol)
        params = tool_def.parameters

        kwargs: dict[str, Any] = {}
        for i, val in enumerate(args):
            if i < len(params):
                kwargs[params[i].name] = val

        original_name = name_map.get(func_symbol, func_symbol)
        calls.append({original_name: kwargs})

    return calls
