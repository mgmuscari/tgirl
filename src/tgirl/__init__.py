"""tgirl: Transformational Grammar for Inference-Restricting Languages."""

from annotated_types import Ge, Gt, Le, Lt, MultipleOf

from tgirl.grammar import (
    GrammarConfig,
    GrammarDiff,
    GrammarOutput,
)
from tgirl.grammar import (
    diff as grammar_diff,
)
from tgirl.grammar import (
    generate as generate_grammar,
)
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
    "GrammarConfig",
    "GrammarDiff",
    "GrammarOutput",
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
    "generate_grammar",
    "grammar_diff",
    "TypeRepr",
    "UnionType",
]
