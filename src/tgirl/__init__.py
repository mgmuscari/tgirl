"""tgirl: Transformational Grammar for Inference-Restricting Languages."""

from annotated_types import Ge, Gt, Le, Lt, MultipleOf

from tgirl.cache import (
    CacheStats,
    make_hf_forward_fn,
    make_mlx_forward_fn,
    make_mlx_forward_fn_torch,
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

try:
    from tgirl.sample_mlx import (
        GrammarStateMlx,
        GrammarTemperatureHookMlx,
        InferenceHookMlx,
        apply_penalties_mlx,
        apply_shaping_mlx,
        run_constrained_generation_mlx,
    )
    from tgirl.transport_mlx import (
        TransportResultMlx,
        redistribute_logits_mlx,
    )
except ImportError:
    pass  # mlx not available

try:
    from tgirl.outlines_adapter import (
        LLGuidanceGrammarStateMlx,
        make_outlines_grammar_factory_mlx,
    )
except ImportError:
    pass  # llguidance/mlx not available

__version__ = "0.1.0"

__all__ = [
    "AnnotatedType",
    "CacheStats",
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
    "make_hf_forward_fn",
    "make_mlx_forward_fn",
    "make_mlx_forward_fn_torch",
    "merge_interventions",
    "redistribute_logits",
    "redistribute_logits_mlx",
    "TransportResultMlx",
    "apply_penalties_mlx",
    "apply_shaping_mlx",
    "GrammarStateMlx",
    "GrammarTemperatureHookMlx",
    "InferenceHookMlx",
    "LLGuidanceGrammarStateMlx",
    "make_outlines_grammar_factory_mlx",
    "run_constrained_generation",
    "run_constrained_generation_mlx",
    "run_pipeline",
    "grammar_diff",
    "TypeRepr",
    "UnionType",
]
