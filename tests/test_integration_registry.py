"""Integration tests for tgirl.registry — full workflow."""

import enum
import json
from typing import Annotated

from annotated_types import Gt, Lt
from pydantic import BaseModel

from tgirl import (
    AnnotatedType,
    ConstraintRepr,
    DictType,
    EnumType,
    ListType,
    ModelType,
    OptionalType,
    PrimitiveType,
    RegistrySnapshot,
    ToolRegistry,
)


class TestFullWorkflow:
    def test_full_workflow_register_and_snapshot(self) -> None:
        reg = ToolRegistry()

        @reg.tool(quota=10, scope="admin")
        def create_user(name: str, age: int) -> str:
            return f"Created {name}"

        @reg.tool(quota=5, scope="user")
        def read_user(user_id: int) -> str:
            return f"User {user_id}"

        @reg.tool(cost=0.1)
        def public_info(query: str) -> str:
            return f"Info: {query}"

        # Snapshot with admin scope
        snap = reg.snapshot(scopes={"admin"})
        names = [t.name for t in snap.tools]
        assert "create_user" in names
        assert "public_info" in names  # scope=None always included
        assert "read_user" not in names  # wrong scope

        # Verify types extracted
        create = next(t for t in snap.tools if t.name == "create_user")
        assert create.parameters[0].name == "name"
        assert create.parameters[0].type_repr == PrimitiveType(kind="str")
        assert create.return_type == PrimitiveType(kind="str")

        # Verify quotas
        assert snap.quotas["create_user"] == 10

    def test_complex_type_signatures(self) -> None:
        reg = ToolRegistry()

        @reg.tool()
        def process(
            data: dict[str, list[int | None]],
        ) -> str:
            return "done"

        snap = reg.snapshot()
        td = snap.tools[0]
        param_type = td.parameters[0].type_repr
        assert isinstance(param_type, DictType)
        assert param_type.key == PrimitiveType(kind="str")
        assert isinstance(param_type.value, ListType)
        inner = param_type.value.element
        assert isinstance(inner, OptionalType)
        assert inner.inner == PrimitiveType(kind="int")

    def test_enum_parameter(self) -> None:
        class Priority(enum.Enum):
            LOW = "low"
            HIGH = "high"

        reg = ToolRegistry()

        @reg.tool()
        def set_priority(p: Priority) -> str:
            return p.value

        snap = reg.snapshot()
        td = snap.tools[0]
        assert isinstance(td.parameters[0].type_repr, EnumType)
        assert td.parameters[0].type_repr.values == ("low", "high")

    def test_annotated_constraints(self) -> None:
        reg = ToolRegistry()

        @reg.tool()
        def bounded(x: Annotated[int, Gt(0), Lt(100)]) -> int:
            return x

        snap = reg.snapshot()
        td = snap.tools[0]
        param = td.parameters[0]
        assert isinstance(param.type_repr, AnnotatedType)
        assert param.type_repr.base == PrimitiveType(kind="int")
        assert param.type_repr.constraints == (
            ConstraintRepr(kind="gt", value=0),
            ConstraintRepr(kind="lt", value=100),
        )

    def test_pydantic_model_parameter(self) -> None:
        class Config(BaseModel):
            name: str
            value: int

        reg = ToolRegistry()

        @reg.tool()
        def apply_config(cfg: Config) -> str:
            return cfg.name

        snap = reg.snapshot()
        td = snap.tools[0]
        param = td.parameters[0]
        assert isinstance(param.type_repr, ModelType)
        assert param.type_repr.name == "Config"
        assert len(param.type_repr.fields) == 2

    def test_snapshot_json_round_trip(self) -> None:
        reg = ToolRegistry()

        @reg.tool(quota=5, scope="admin")
        def greet(name: str) -> str:
            return name

        snap = reg.snapshot(scopes={"admin"})
        json_str = snap.model_dump_json()
        restored = RegistrySnapshot.model_validate_json(json_str)

        assert restored.tools == snap.tools
        assert dict(restored.quotas) == dict(snap.quotas)
        assert restored.cost_remaining == snap.cost_remaining
        assert restored.scopes == snap.scopes
        assert restored.timestamp == snap.timestamp

    def test_snapshot_json_schema(self) -> None:
        schema = RegistrySnapshot.model_json_schema()
        assert isinstance(schema, dict)
        json_str = json.dumps(schema)
        assert "tools" in json_str
        assert "quotas" in json_str

    def test_imports_from_package(self) -> None:
        """All public types are importable from tgirl package root."""
        from tgirl import (  # noqa: F401
            AnnotatedType,
            ConstraintRepr,
            DictType,
            EnumType,
            Gt,
            ListType,
            Lt,
            ModelType,
            OptionalType,
            PrimitiveType,
            RegistrySnapshot,
            ToolRegistry,
        )
