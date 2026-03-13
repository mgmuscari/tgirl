"""Shared type definitions for tgirl.

Frozen Pydantic models for type representation, tool definitions,
registry snapshots, pipeline errors, and telemetry records.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

# --- Type Representation System ---


class PrimitiveType(BaseModel):
    """Primitive Python types: str, int, float, bool, none."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["str", "int", "float", "bool", "none"]
    type_tag: Literal["primitive"] = "primitive"


class ListType(BaseModel):
    """Homogeneous list type with element type."""

    model_config = ConfigDict(frozen=True)
    element: TypeRepr
    type_tag: Literal["list"] = "list"


class DictType(BaseModel):
    """Dict type with key and value types."""

    model_config = ConfigDict(frozen=True)
    key: TypeRepr
    value: TypeRepr
    type_tag: Literal["dict"] = "dict"


class LiteralType(BaseModel):
    """Literal type with enumerated values."""

    model_config = ConfigDict(frozen=True)
    values: tuple[str | int | float | bool, ...]
    type_tag: Literal["literal"] = "literal"


class EnumType(BaseModel):
    """Enum type with name and string values."""

    model_config = ConfigDict(frozen=True)
    name: str
    values: tuple[str, ...]
    type_tag: Literal["enum"] = "enum"


class OptionalType(BaseModel):
    """Optional type wrapping an inner type."""

    model_config = ConfigDict(frozen=True)
    inner: TypeRepr
    type_tag: Literal["optional"] = "optional"


class UnionType(BaseModel):
    """Union of multiple types."""

    model_config = ConfigDict(frozen=True)
    members: tuple[TypeRepr, ...]
    type_tag: Literal["union"] = "union"


class ConstraintRepr(BaseModel):
    """Serializable representation of an annotated-types constraint."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["gt", "lt", "ge", "le", "multiple_of"]
    value: int | float


class FieldDef(BaseModel):
    """Field definition for model types."""

    model_config = ConfigDict(frozen=True)
    name: str
    type_repr: TypeRepr
    required: bool
    default: Any = None


class ModelType(BaseModel):
    """Pydantic/dataclass model type with named fields."""

    model_config = ConfigDict(frozen=True)
    name: str
    fields: tuple[FieldDef, ...]
    type_tag: Literal["model"] = "model"


class AnnotatedType(BaseModel):
    """Type with annotated-types constraints."""

    model_config = ConfigDict(frozen=True)
    base: TypeRepr
    constraints: tuple[ConstraintRepr, ...]
    type_tag: Literal["annotated"] = "annotated"


class AnyType(BaseModel):
    """Represents typing.Any -- unconstrained type.

    Grammar module decides production semantics.
    """

    model_config = ConfigDict(frozen=True)
    type_tag: Literal["any"] = "any"


TypeRepr = Annotated[
    PrimitiveType
    | ListType
    | DictType
    | LiteralType
    | EnumType
    | OptionalType
    | UnionType
    | ModelType
    | AnnotatedType
    | AnyType,
    Field(discriminator="type_tag"),
]


# --- Parameter and Tool Definitions ---


class ParameterDef(BaseModel):
    """Function parameter definition."""

    model_config = ConfigDict(frozen=True)
    name: str
    type_repr: TypeRepr
    default: Any = None
    has_default: bool = False


class ToolDefinition(BaseModel):
    """Registered tool definition per spec section 3.3."""

    model_config = ConfigDict(frozen=True)
    name: str
    parameters: tuple[ParameterDef, ...]
    return_type: TypeRepr
    quota: int | None = None
    cost: float = 0.0
    cost_budget: float | None = None
    scope: str | None = None
    timeout: float | None = None
    cacheable: bool = False
    description: str = ""
    param_tags: tuple[tuple[str, str], ...] = ()
    examples: tuple[str, ...] = ()


# --- Registry Snapshot ---


class RegistrySnapshot(BaseModel):
    """Immutable point-in-time snapshot of registry state per spec section 3.3.

    The quotas field uses MappingProxyType to enforce deep immutability.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    tools: tuple[ToolDefinition, ...]
    quotas: Mapping[str, int]
    cost_remaining: float | None
    scopes: frozenset[str]
    timestamp: float
    type_grammars: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _wrap_quotas(self) -> RegistrySnapshot:
        if not isinstance(self.quotas, MappingProxyType):
            object.__setattr__(self, "quotas", MappingProxyType(dict(self.quotas)))
        return self

    @field_serializer("quotas")
    @classmethod
    def _serialize_quotas(cls, v: Mapping[str, int]) -> dict[str, int]:
        return dict(v)


# --- Pipeline Error ---


class PipelineError(BaseModel):
    """Pipeline execution error per spec section 5.7."""

    model_config = ConfigDict(frozen=True)
    stage: str
    tool_name: str | None = None
    error_type: str
    message: str
    hy_source: str
    position: int | None = None


# --- Telemetry Record (stub) ---


class TelemetryRecord(BaseModel):
    """Telemetry record per spec section 8.6."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    pipeline_id: str
    tokens: list[int]
    grammar_valid_counts: list[int]
    temperatures_applied: list[float]
    wasserstein_distances: list[float]
    top_p_applied: list[float]
    token_log_probs: list[float]
    grammar_generation_ms: float
    ot_computation_total_ms: float
    ot_bypassed_count: int
    hy_source: str
    execution_result: Any = None
    execution_error: PipelineError | None = None
    cycle_number: int
    freeform_tokens_before: int
    wall_time_ms: float
    total_tokens: int
    model_id: str
    registry_snapshot_hash: str
    rerank_selected_tool: str | None = None
    rerank_routing_tokens: int | None = None
    rerank_latency_ms: float | None = None


# --- Reranking ---


class RerankConfig(BaseModel):
    """Configuration for grammar-constrained tool re-ranking."""

    model_config = ConfigDict(frozen=True)
    max_tokens: int = 16
    temperature: float = 0.3
    top_k: int = 1
    enabled: bool = True


class RerankResult(BaseModel):
    """Result of a re-ranking pass."""

    model_config = ConfigDict(frozen=True)
    selected_tools: tuple[str, ...]
    routing_tokens: int
    routing_latency_ms: float
    routing_grammar_text: str


# --- Model Intervention ---


class ModelIntervention(BaseModel):
    """Per-token model intervention from inference hooks."""

    model_config = ConfigDict(frozen=True)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    activation_steering: Any | None = None  # Reserved for ESTRADIOL


# --- Session Config ---


class SessionConfig(BaseModel):
    """Configuration for a dual-mode sampling session."""

    model_config = ConfigDict(frozen=True)
    # Freeform mode
    freeform_temperature: float = 0.7
    freeform_top_p: float = 0.9
    freeform_top_k: int | None = None
    freeform_repetition_penalty: float = 1.0
    freeform_max_tokens: int = 4096
    # Constrained mode
    constrained_base_temperature: float = 0.3
    constrained_ot_epsilon: float = 0.1
    constrained_max_tokens: int = 512
    # Session-level
    max_tool_cycles: int = 10
    session_cost_budget: float | None = None
    session_timeout: float = 300.0
    # Delimiters
    tool_open_delimiter: str = "<tool>"
    tool_close_delimiter: str = "</tool>"
    result_open_delimiter: str = "<tool_result>"
    result_close_delimiter: str = "</tool_result>"


# Rebuild forward refs for recursive types
ListType.model_rebuild()
DictType.model_rebuild()
OptionalType.model_rebuild()
UnionType.model_rebuild()
ModelType.model_rebuild()
AnnotatedType.model_rebuild()
