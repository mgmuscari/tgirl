"""Tests for tgirl.types — type representation system."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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
    RegistrySnapshot,
    ToolDefinition,
    TypeRepr,
    UnionType,
)


class TestPrimitiveType:
    def test_primitive_types_are_frozen(self) -> None:
        p = PrimitiveType(kind="str")
        with pytest.raises(ValidationError):
            p.kind = "int"  # type: ignore[misc]

    @pytest.mark.parametrize("kind", ["str", "int", "float", "bool", "none"])
    def test_primitive_kinds(self, kind: str) -> None:
        p = PrimitiveType(kind=kind)  # type: ignore[arg-type]
        assert p.kind == kind


class TestListType:
    def test_list_type_nests_correctly(self) -> None:
        lt = ListType(element=PrimitiveType(kind="str"))
        assert lt.element == PrimitiveType(kind="str")
        assert lt.type_tag == "list"


class TestDictType:
    def test_dict_type_nests_correctly(self) -> None:
        dt = DictType(
            key=PrimitiveType(kind="str"),
            value=PrimitiveType(kind="int"),
        )
        assert dt.key == PrimitiveType(kind="str")
        assert dt.value == PrimitiveType(kind="int")


class TestUnionType:
    def test_union_type_preserves_members(self) -> None:
        members = (PrimitiveType(kind="str"), PrimitiveType(kind="int"))
        ut = UnionType(members=members)
        assert ut.members == members


class TestOptionalType:
    def test_optional_type_wraps_inner(self) -> None:
        ot = OptionalType(inner=PrimitiveType(kind="str"))
        assert ot.inner == PrimitiveType(kind="str")


class TestLiteralType:
    def test_literal_type_preserves_values(self) -> None:
        lt = LiteralType(values=("a", "b", 1))
        assert lt.values == ("a", "b", 1)


class TestModelType:
    def test_model_type_holds_fields(self) -> None:
        fields = (
            FieldDef(
                name="x",
                type_repr=PrimitiveType(kind="int"),
                required=True,
                default=None,
            ),
        )
        mt = ModelType(name="Point", fields=fields)
        assert mt.fields == fields
        assert mt.name == "Point"


class TestAnnotatedType:
    def test_annotated_type_holds_constraints(self) -> None:
        constraints = (
            ConstraintRepr(kind="gt", value=0),
            ConstraintRepr(kind="lt", value=100),
        )
        at = AnnotatedType(
            base=PrimitiveType(kind="int"),
            constraints=constraints,
        )
        assert at.constraints == constraints
        assert at.base == PrimitiveType(kind="int")


class TestAnyType:
    def test_any_type_is_frozen(self) -> None:
        a = AnyType()
        with pytest.raises(ValidationError):
            a.type_tag = "something"  # type: ignore[misc]

    def test_any_type_tag(self) -> None:
        a = AnyType()
        assert a.type_tag == "any"


class TestTypeReprDiscriminatedUnion:
    def test_type_repr_discriminated_union(self) -> None:
        """Pydantic correctly deserializes each variant via type_tag."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(TypeRepr)

        cases = [
            {"kind": "str", "type_tag": "primitive"},
            {"element": {"kind": "int", "type_tag": "primitive"}, "type_tag": "list"},
            {
                "key": {"kind": "str", "type_tag": "primitive"},
                "value": {"kind": "int", "type_tag": "primitive"},
                "type_tag": "dict",
            },
            {"values": ["a", "b"], "type_tag": "literal"},
            {"name": "Color", "values": ["red", "blue"], "type_tag": "enum"},
            {
                "inner": {"kind": "str", "type_tag": "primitive"},
                "type_tag": "optional",
            },
            {
                "members": [
                    {"kind": "str", "type_tag": "primitive"},
                    {"kind": "int", "type_tag": "primitive"},
                ],
                "type_tag": "union",
            },
            {"name": "M", "fields": [], "type_tag": "model"},
            {
                "base": {"kind": "int", "type_tag": "primitive"},
                "constraints": [{"kind": "gt", "value": 0}],
                "type_tag": "annotated",
            },
            {"type_tag": "any"},
        ]
        expected_types = [
            PrimitiveType,
            ListType,
            DictType,
            LiteralType,
            EnumType,
            OptionalType,
            UnionType,
            ModelType,
            AnnotatedType,
            AnyType,
        ]
        for data, expected in zip(cases, expected_types, strict=True):
            result = adapter.validate_python(data)
            assert isinstance(result, expected), (
                f"Expected {expected.__name__}, got {type(result).__name__}"
            )

    def test_type_repr_json_round_trip(self) -> None:
        """model_dump_json -> model_validate_json round-trips for all variants."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(TypeRepr)
        variants: list[TypeRepr] = [
            PrimitiveType(kind="str"),
            ListType(element=PrimitiveType(kind="int")),
            DictType(
                key=PrimitiveType(kind="str"),
                value=PrimitiveType(kind="float"),
            ),
            LiteralType(values=("x", "y")),
            EnumType(name="Color", values=("red", "blue")),
            OptionalType(inner=PrimitiveType(kind="bool")),
            UnionType(
                members=(PrimitiveType(kind="str"), PrimitiveType(kind="int"))
            ),
            ModelType(name="Empty", fields=()),
            AnnotatedType(
                base=PrimitiveType(kind="int"),
                constraints=(ConstraintRepr(kind="gt", value=0),),
            ),
            AnyType(),
        ]
        for variant in variants:
            json_str = adapter.dump_json(variant)
            restored = adapter.validate_json(json_str)
            assert restored == variant, (
                f"Round-trip failed for {type(variant).__name__}"
            )

    def test_type_repr_json_schema(self) -> None:
        """model_json_schema produces valid schema for discriminated union."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(TypeRepr)
        schema = adapter.json_schema()
        assert isinstance(schema, dict)
        # Should have discriminator info
        json_str = json.dumps(schema)
        assert "type_tag" in json_str

    def test_type_repr_equality(self) -> None:
        a = PrimitiveType(kind="str")
        b = PrimitiveType(kind="str")
        assert a == b

    def test_type_repr_hashable(self) -> None:
        variants: list[TypeRepr] = [
            PrimitiveType(kind="str"),
            ListType(element=PrimitiveType(kind="int")),
            AnyType(),
        ]
        s = set()
        for v in variants:
            s.add(v)
        assert len(s) == 3


class TestParameterDef:
    def test_parameter_def_frozen(self) -> None:
        pd = ParameterDef(
            name="x",
            type_repr=PrimitiveType(kind="int"),
            default=None,
            has_default=False,
        )
        with pytest.raises(ValidationError):
            pd.name = "y"  # type: ignore[misc]


class TestToolDefinition:
    def test_tool_definition_frozen(self) -> None:
        td = ToolDefinition(
            name="test_tool",
            parameters=(),
            return_type=PrimitiveType(kind="str"),
            quota=None,
            cost=0.0,
            cost_budget=None,
            scope=None,
            timeout=None,
            cacheable=False,
            description="A test tool",
        )
        with pytest.raises(ValidationError):
            td.name = "other"  # type: ignore[misc]

    def test_tool_definition_json_schema(self) -> None:
        schema = ToolDefinition.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


class TestRegistrySnapshot:
    def test_registry_snapshot_frozen(self) -> None:
        snap = RegistrySnapshot(
            tools=(),
            quotas={},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=0.0,
        )
        with pytest.raises(ValidationError):
            snap.timestamp = 1.0  # type: ignore[misc]

    def test_registry_snapshot_quotas_immutable(self) -> None:
        snap = RegistrySnapshot(
            tools=(),
            quotas={"tool_a": 5},
            cost_remaining=None,
            scopes=frozenset(),
            timestamp=0.0,
        )
        with pytest.raises(TypeError):
            snap.quotas["tool_a"] = 999  # type: ignore[index]
        with pytest.raises(TypeError):
            snap.quotas["new_tool"] = 1  # type: ignore[index]


class TestRerankConfig:
    def test_rerank_config_defaults(self) -> None:
        from tgirl.types import RerankConfig

        cfg = RerankConfig()
        assert cfg.max_tokens == 16
        assert cfg.temperature == 0.3
        assert cfg.top_k == 1
        assert cfg.enabled is True

    def test_rerank_config_is_frozen(self) -> None:
        from tgirl.types import RerankConfig

        cfg = RerankConfig()
        with pytest.raises(ValidationError):
            cfg.max_tokens = 32  # type: ignore[misc]

    def test_rerank_config_custom_values(self) -> None:
        from tgirl.types import RerankConfig

        cfg = RerankConfig(max_tokens=8, temperature=0.5, top_k=2, enabled=False)
        assert cfg.max_tokens == 8
        assert cfg.temperature == 0.5
        assert cfg.top_k == 2
        assert cfg.enabled is False


class TestRerankResult:
    def test_rerank_result_constructs(self) -> None:
        from tgirl.types import RerankResult

        result = RerankResult(
            selected_tools=("get_field",),
            routing_tokens=3,
            routing_latency_ms=12.5,
            routing_grammar_text='start: tool_choice\ntool_choice: "get_field"\n',
        )
        assert result.selected_tools == ("get_field",)
        assert result.routing_tokens == 3
        assert result.routing_latency_ms == 12.5

    def test_rerank_result_is_frozen(self) -> None:
        from tgirl.types import RerankResult

        result = RerankResult(
            selected_tools=("get_field",),
            routing_tokens=3,
            routing_latency_ms=12.5,
            routing_grammar_text='start: tool_choice\ntool_choice: "get_field"\n',
        )
        with pytest.raises(ValidationError):
            result.routing_tokens = 10  # type: ignore[misc]


class TestTelemetryRecordRerank:
    def test_telemetry_record_rerank_fields_default_none(self) -> None:
        from tgirl.types import TelemetryRecord

        record = TelemetryRecord(
            pipeline_id="test",
            tokens=[1, 2],
            grammar_valid_counts=[10, 5],
            temperatures_applied=[0.3, 0.3],
            wasserstein_distances=[0.1, 0.2],
            top_p_applied=[-1.0, -1.0],
            token_log_probs=[-0.5, -0.3],
            grammar_generation_ms=5.0,
            ot_computation_total_ms=2.0,
            ot_bypassed_count=0,
            hy_source="(tool 1)",
            cycle_number=1,
            freeform_tokens_before=10,
            wall_time_ms=100.0,
            total_tokens=12,
            model_id="test-model",
            registry_snapshot_hash="abc123",
        )
        assert record.rerank_selected_tool is None
        assert record.rerank_routing_tokens is None
        assert record.rerank_latency_ms is None

    def test_telemetry_record_rerank_fields_populated(self) -> None:
        from tgirl.types import TelemetryRecord

        record = TelemetryRecord(
            pipeline_id="test",
            tokens=[1, 2],
            grammar_valid_counts=[10, 5],
            temperatures_applied=[0.3, 0.3],
            wasserstein_distances=[0.1, 0.2],
            top_p_applied=[-1.0, -1.0],
            token_log_probs=[-0.5, -0.3],
            grammar_generation_ms=5.0,
            ot_computation_total_ms=2.0,
            ot_bypassed_count=0,
            hy_source="(tool 1)",
            cycle_number=1,
            freeform_tokens_before=10,
            wall_time_ms=100.0,
            total_tokens=12,
            model_id="test-model",
            registry_snapshot_hash="abc123",
            rerank_selected_tool="get_field",
            rerank_routing_tokens=3,
            rerank_latency_ms=12.5,
        )
        assert record.rerank_selected_tool == "get_field"
        assert record.rerank_routing_tokens == 3
        assert record.rerank_latency_ms == 12.5
