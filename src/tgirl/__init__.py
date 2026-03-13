"""tgirl: Transformational Grammar for Inference-Restricting Languages."""

from annotated_types import Ge, Gt, Le, Lt, MultipleOf

from tgirl.compile import (
    CompileConfig,
    InsufficientResources,
    PipelineResult,
    run_pipeline,
)
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
from tgirl.sample import (
    ConstrainedGenerationResult,
    DelimiterDetector,
    GrammarState,
    GrammarTemperatureHook,
    InferenceHook,
    SamplingResult,
    SamplingSession,
    ToolCallRecord,
    apply_penalties,
    apply_shaping,
    merge_interventions,
    run_constrained_generation,
)
from tgirl.transport import (
    TransportConfig,
    TransportResult,
    redistribute_logits,
)
from tgirl.types import (
    AnnotatedType,
    AnyType,
    ConstraintRepr,
    DictType,
    EnumType,
    FieldDef,
    ListType,
    LiteralType,
    ModelIntervention,
    ModelType,
    OptionalType,
    ParameterDef,
    PipelineError,
    PrimitiveType,
    RegistrySnapshot,
    SessionConfig,
    TelemetryRecord,
    ToolDefinition,
    TypeRepr,
    UnionType,
)

__version__ = "0.1.0"

__all__ = [
    "AnnotatedType",
    "CompileConfig",
    "ConstrainedGenerationResult",
    "AnyType",
    "ConstraintRepr",
    "DelimiterDetector",
    "DictType",
    "EnumType",
    "FieldDef",
    "Ge",
    "GrammarConfig",
    "GrammarState",
    "GrammarTemperatureHook",
    "InferenceHook",
    "InsufficientResources",
    "GrammarDiff",
    "GrammarOutput",
    "Gt",
    "Le",
    "ListType",
    "LiteralType",
    "Lt",
    "ModelIntervention",
    "ModelType",
    "MultipleOf",
    "OptionalType",
    "ParameterDef",
    "PipelineError",
    "PipelineResult",
    "PrimitiveType",
    "RegistrySnapshot",
    "SamplingResult",
    "SamplingSession",
    "SessionConfig",
    "TelemetryRecord",
    "ToolCallRecord",
    "ToolDefinition",
    "ToolRegistry",
    "TransportConfig",
    "TransportResult",
    "apply_penalties",
    "apply_shaping",
    "generate_grammar",
    "merge_interventions",
    "redistribute_logits",
    "run_constrained_generation",
    "run_pipeline",
    "grammar_diff",
    "TypeRepr",
    "UnionType",
]
