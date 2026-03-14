"""tgirl: Transformational Grammar for Inference-Restricting Languages."""

from annotated_types import Ge, Gt, Le, Lt, MultipleOf

from tgirl.bfcl import (
    load_ground_truth,
    load_test_data,
    register_bfcl_tools,
    sexpr_to_bfcl,
)
from tgirl.compile import (
    CompileConfig,
    InsufficientResources,
    PipelineResult,
    run_pipeline,
)
from tgirl.format import (
    ChatTemplateFormatter,
    PlainFormatter,
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
from tgirl.rerank import ToolRouter
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
    PromptFormatter,
    RegistrySnapshot,
    RerankConfig,
    RerankResult,
    SessionConfig,
    TelemetryRecord,
    ToolDefinition,
    TypeRepr,
    UnionType,
)

__version__ = "0.1.0"

__all__ = [
    "AnnotatedType",
    "load_ground_truth",
    "load_test_data",
    "register_bfcl_tools",
    "sexpr_to_bfcl",
    "ChatTemplateFormatter",
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
    "PlainFormatter",
    "PrimitiveType",
    "PromptFormatter",
    "RegistrySnapshot",
    "RerankConfig",
    "RerankResult",
    "SamplingResult",
    "SamplingSession",
    "SessionConfig",
    "TelemetryRecord",
    "ToolCallRecord",
    "ToolDefinition",
    "ToolRegistry",
    "ToolRouter",
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
