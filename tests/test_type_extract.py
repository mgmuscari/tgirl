"""Tests for tgirl._type_extract — type extraction engine."""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal

import pytest
from annotated_types import Ge, Gt, Le, Lt, MultipleOf
from pydantic import BaseModel

from tgirl._type_extract import extract_parameters, extract_type
from tgirl.types import (
    AnnotatedType,
    AnyType,
    ConstraintRepr,
    DictType,
    EnumType,
    ListType,
    LiteralType,
    ModelType,
    OptionalType,
    PrimitiveType,
    UnionType,
)


class TestExtractPrimitives:
    def test_extract_str(self) -> None:
        assert extract_type(str) == PrimitiveType(kind="str")

    def test_extract_int(self) -> None:
        assert extract_type(int) == PrimitiveType(kind="int")

    def test_extract_float(self) -> None:
        assert extract_type(float) == PrimitiveType(kind="float")

    def test_extract_bool(self) -> None:
        assert extract_type(bool) == PrimitiveType(kind="bool")

    def test_extract_none(self) -> None:
        assert extract_type(type(None)) == PrimitiveType(kind="none")


class TestExtractAny:
    def test_extract_any(self) -> None:
        assert extract_type(Any) == AnyType()

    def test_extract_dict_str_any(self) -> None:
        result = extract_type(dict[str, Any])
        assert isinstance(result, DictType)
        assert result.key == PrimitiveType(kind="str")
        assert result.value == AnyType()


class TestExtractContainers:
    def test_extract_list_str(self) -> None:
        result = extract_type(list[str])
        assert result == ListType(element=PrimitiveType(kind="str"))

    def test_extract_list_nested(self) -> None:
        result = extract_type(list[list[int]])
        assert result == ListType(
            element=ListType(element=PrimitiveType(kind="int"))
        )

    def test_extract_dict(self) -> None:
        result = extract_type(dict[str, int])
        assert result == DictType(
            key=PrimitiveType(kind="str"),
            value=PrimitiveType(kind="int"),
        )


class TestExtractLiteral:
    def test_extract_literal_strings(self) -> None:
        result = extract_type(Literal["a", "b"])
        assert isinstance(result, LiteralType)
        assert result.values == ("a", "b")

    def test_extract_literal_ints(self) -> None:
        result = extract_type(Literal[1, 2, 3])
        assert isinstance(result, LiteralType)
        assert result.values == (1, 2, 3)


class TestExtractOptionalUnion:
    def test_extract_optional(self) -> None:
        result = extract_type(str | None)
        assert isinstance(result, OptionalType)
        assert result.inner == PrimitiveType(kind="str")

    def test_extract_union(self) -> None:
        result = extract_type(str | int)
        assert isinstance(result, UnionType)
        assert result.members == (
            PrimitiveType(kind="str"),
            PrimitiveType(kind="int"),
        )


class TestExtractEnum:
    def test_extract_enum(self) -> None:
        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        result = extract_type(Color)
        assert isinstance(result, EnumType)
        assert result.name == "Color"
        assert result.values == ("red", "blue")


class TestExtractAnnotated:
    def test_extract_annotated_with_gt_lt(self) -> None:
        result = extract_type(Annotated[int, Gt(0), Lt(100)])
        assert isinstance(result, AnnotatedType)
        assert result.base == PrimitiveType(kind="int")
        assert result.constraints == (
            ConstraintRepr(kind="gt", value=0),
            ConstraintRepr(kind="lt", value=100),
        )

    def test_extract_annotated_all_constraint_types(self) -> None:
        result = extract_type(
            Annotated[float, Gt(0), Lt(100), Ge(1), Le(99), MultipleOf(5)]
        )
        assert isinstance(result, AnnotatedType)
        assert len(result.constraints) == 5

    def test_extract_annotated_without_constraints(self) -> None:
        """Non-constraint metadata should not produce AnnotatedType."""
        result = extract_type(Annotated[str, "some metadata"])
        assert result == PrimitiveType(kind="str")


class TestExtractPydanticModel:
    def test_extract_pydantic_model(self) -> None:
        class Point(BaseModel):
            x: int
            y: int

        result = extract_type(Point)
        assert isinstance(result, ModelType)
        assert result.name == "Point"
        assert len(result.fields) == 2
        assert result.fields[0].name == "x"
        assert result.fields[0].type_repr == PrimitiveType(kind="int")
        assert result.fields[0].required is True

    def test_extract_pydantic_optional_field(self) -> None:
        class Item(BaseModel):
            name: str
            tag: str | None = None

        result = extract_type(Item)
        assert isinstance(result, ModelType)
        tag_field = result.fields[1]
        assert tag_field.name == "tag"
        assert isinstance(tag_field.type_repr, OptionalType)
        assert tag_field.required is False

    def test_extract_pydantic_nested_model(self) -> None:
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner

        result = extract_type(Outer)
        assert isinstance(result, ModelType)
        inner_field = result.fields[0]
        assert isinstance(inner_field.type_repr, ModelType)
        assert inner_field.type_repr.name == "Inner"


class TestExtractUnsupported:
    def test_extract_unsupported_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported type"):
            extract_type(bytes)


class TestExtractParameters:
    def test_extract_parameters_simple_function(self) -> None:
        def greet(name: str, count: int) -> str:
            return name * count

        params = extract_parameters(greet)
        assert len(params) == 2
        assert params[0].name == "name"
        assert params[0].type_repr == PrimitiveType(kind="str")
        assert params[0].has_default is False
        assert params[1].name == "count"
        assert params[1].type_repr == PrimitiveType(kind="int")

    def test_extract_parameters_with_defaults(self) -> None:
        def greet(name: str, count: int = 1) -> str:
            return name * count

        params = extract_parameters(greet)
        assert params[1].has_default is True
        assert params[1].default == 1

    def test_extract_parameters_missing_annotation_raises(self) -> None:
        def bad(x) -> str:  # type: ignore[no-untyped-def]
            return str(x)

        with pytest.raises(TypeError, match="missing type annotation"):
            extract_parameters(bad)

    def test_extract_return_type(self) -> None:
        def greet(name: str) -> str:
            return name

        _, return_type = extract_parameters(greet, include_return=True)
        assert return_type == PrimitiveType(kind="str")

    def test_extract_parameters_missing_return_type_raises(self) -> None:
        def bad(x: str):  # type: ignore[no-untyped-def]
            return x

        with pytest.raises(TypeError, match="missing return type"):
            extract_parameters(bad, include_return=True)
