"""Type extraction engine for converting Python annotations to TypeRepr.

Private module — downstream consumers use TypeRepr from snapshots,
not this extraction API directly.
"""

from __future__ import annotations

import enum
import inspect
import types
import typing
from typing import Any, Literal, overload

from annotated_types import Ge, Gt, Le, Lt, MultipleOf
from pydantic import BaseModel

from tgirl.types import (
    AnnotatedType,
    AnyType,
    ConstraintRepr,
    DictType,
    EnumType,
    FieldDef,
    ListType,
    LiteralType,
    ModelType,
    OptionalType,
    ParameterDef,
    PrimitiveType,
    TypeRepr,
    UnionType,
)

_PrimitiveKind = Literal["str", "int", "float", "bool", "none"]
_ConstraintKind = Literal["gt", "lt", "ge", "le", "multiple_of"]

_PRIMITIVE_MAP: dict[type, _PrimitiveKind] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    type(None): "none",
}

_CONSTRAINT_MAP: dict[type, _ConstraintKind] = {
    Gt: "gt",
    Lt: "lt",
    Ge: "ge",
    Le: "le",
    MultipleOf: "multiple_of",
}


def _extract_constraint_value(
    meta: Any, kind_name: str
) -> int | float:
    """Extract the numeric value from an annotated-types constraint."""
    attr_map = {
        "gt": "gt",
        "lt": "lt",
        "ge": "ge",
        "le": "le",
        "multiple_of": "multiple_of",
    }
    return getattr(meta, attr_map[kind_name])  # type: ignore[no-any-return]


def extract_type(annotation: Any) -> TypeRepr:
    """Convert a Python type annotation to a TypeRepr Pydantic model.

    Args:
        annotation: A Python type annotation (e.g., str, list[int],
            Optional[str], Annotated[int, Gt(0)]).

    Returns:
        The corresponding TypeRepr variant.

    Raises:
        TypeError: If the annotation is not a supported type.
    """
    # typing.Any
    if annotation is Any:
        return AnyType()

    # Primitives (check bool before int since bool is a subclass of int)
    if annotation in _PRIMITIVE_MAP:
        return PrimitiveType(kind=_PRIMITIVE_MAP[annotation])

    # Enum subclass (check before other type checks)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = tuple(member.value for member in annotation)
        return EnumType(name=annotation.__name__, values=values)

    # Pydantic BaseModel subclass
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields: list[FieldDef] = []
        for name, field_info in annotation.model_fields.items():
            field_annotation = field_info.annotation
            if field_annotation is None:
                msg = f"Field {name} on {annotation.__name__} has no annotation"
                raise TypeError(msg)
            field_type = extract_type(field_annotation)
            required = field_info.is_required()
            default = field_info.default if not required else None
            fields.append(
                FieldDef(
                    name=name,
                    type_repr=field_type,
                    required=required,
                    default=default,
                )
            )
        return ModelType(name=annotation.__name__, fields=tuple(fields))

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Annotated[T, ...]
    if origin is typing.Annotated:
        base_type = extract_type(args[0])
        constraints: list[ConstraintRepr] = []
        for meta in args[1:]:
            for constraint_cls, kind_name in _CONSTRAINT_MAP.items():
                if isinstance(meta, constraint_cls):
                    value = _extract_constraint_value(
                        meta, kind_name
                    )
                    constraints.append(
                        ConstraintRepr(
                            kind=kind_name, value=value
                        )
                    )
                    break
        if constraints:
            return AnnotatedType(
                base=base_type, constraints=tuple(constraints)
            )
        return base_type

    # list[T]
    if origin is list:
        if not args:
            return ListType(element=AnyType())
        return ListType(element=extract_type(args[0]))

    # dict[K, V]
    if origin is dict:
        if not args:
            return DictType(key=AnyType(), value=AnyType())
        return DictType(
            key=extract_type(args[0]),
            value=extract_type(args[1]),
        )

    # Literal[...]
    if origin is typing.Literal:
        return LiteralType(values=args)

    # Union (includes Optional) — both typing.Union and X | Y syntax
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        if isinstance(annotation, types.UnionType):
            args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            # Optional[T] = Union[T, None]
            return OptionalType(inner=extract_type(non_none[0]))
        return UnionType(
            members=tuple(extract_type(a) for a in args)
        )

    msg = f"Unsupported type: {annotation}"
    raise TypeError(msg)


@overload
def extract_parameters(
    func: Any,
    *,
    include_return: typing.Literal[False] = ...,
) -> tuple[ParameterDef, ...]: ...


@overload
def extract_parameters(
    func: Any,
    *,
    include_return: typing.Literal[True],
) -> tuple[tuple[ParameterDef, ...], TypeRepr]: ...


def extract_parameters(
    func: Any,
    *,
    include_return: bool = False,
) -> tuple[ParameterDef, ...] | tuple[tuple[ParameterDef, ...], TypeRepr]:
    """Extract parameter definitions from a function signature.

    Args:
        func: The function to extract parameters from.
        include_return: If True, also extract and return the return type.

    Returns:
        A tuple of ParameterDef, or (params, return_type) if include_return.

    Raises:
        TypeError: If any parameter lacks a type annotation, or if
            include_return is True and the function lacks a return type.
    """
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func, include_extras=True)

    params: list[ParameterDef] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if name not in hints:
            msg = f"Parameter '{name}' is missing type annotation"
            raise TypeError(msg)
        type_repr = extract_type(hints[name])
        has_default = param.default is not inspect.Parameter.empty
        default = param.default if has_default else None
        params.append(
            ParameterDef(
                name=name,
                type_repr=type_repr,
                default=default,
                has_default=has_default,
            )
        )

    result_params = tuple(params)

    if include_return:
        if "return" not in hints:
            msg = f"Function '{func.__name__}' is missing return type annotation"
            raise TypeError(msg)
        return_type = extract_type(hints["return"])
        return result_params, return_type

    return result_params
