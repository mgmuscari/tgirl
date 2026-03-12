"""tgirl: Transformational Grammar for Inference-Restricting Languages."""

from annotated_types import Ge, Gt, Le, Lt, MultipleOf

from tgirl.registry import ToolRegistry
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
    PipelineError,
    PrimitiveType,
    RegistrySnapshot,
    TelemetryRecord,
    ToolDefinition,
    TypeRepr,
    UnionType,
)

__version__ = "0.1.0"

__all__ = [
    "AnnotatedType",
    "AnyType",
    "ConstraintRepr",
    "DictType",
    "EnumType",
    "FieldDef",
    "Ge",
    "Gt",
    "Le",
    "ListType",
    "LiteralType",
    "Lt",
    "ModelType",
    "MultipleOf",
    "OptionalType",
    "ParameterDef",
    "PipelineError",
    "PrimitiveType",
    "RegistrySnapshot",
    "TelemetryRecord",
    "ToolDefinition",
    "ToolRegistry",
    "TypeRepr",
    "UnionType",
]
