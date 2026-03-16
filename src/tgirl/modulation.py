"""ADSR modulation matrix for constrained generation.

Routes source signals through a configurable (11, 7) matrix to produce
per-token sampling parameter modulations. Replaces GrammarTemperatureHook
and NestingDepthHook with a single unified hook.

Design: docs/design/adsr-envelope.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tgirl.types import ModelIntervention

if TYPE_CHECKING:
    import mlx.core as mx
    import torch

    from tgirl.sample import GrammarState


@dataclass(frozen=True)
class SourceConditionerConfig:
    """Per-source preprocessing config."""

    range_min: float = 0.0
    range_max: float = 1.0
    rectify: bool = False
    invert: bool = False
    slew_rate: float = 1.0  # 1.0 = no smoothing, 0.1 = heavy smoothing


@dataclass
class EnvelopeState:
    """Mutable per-generation state for phase detection."""

    phase: str = "attack"  # attack | decay | sustain | release
    phase_position: int = 0
    depth: int = 0
    prev_depth: int = 0
    prev_freedom: float = 1.0
    peak_freedom: float = 0.0
    prev_smoothed: list[float] = field(default_factory=lambda: [0.0] * 11)
    # Hysteresis state
    pending_phase: str | None = None  # candidate phase waiting for confirmation
    pending_count: int = 0  # consecutive tokens meeting transition condition
    min_phase_duration: int = 2  # minimum tokens in a phase before transition allowed

    def detect_phase(self, freedom: float, depth: int) -> str:
        """Detect ADSR phase from structural signals with hysteresis.

        Transition rules:
        - attack: depth increased. Immediate (no hysteresis) -- each opening
          paren genuinely starts a new tool selection regime.
        - decay: freedom collapsed >70% from peak while in attack.
          Requires 2 consecutive tokens meeting condition (hysteresis).
        - sustain: been in decay for >2 tokens (existing time gate).
          No additional hysteresis needed.
        - release: depth decreased AND depth <= 1 (approaching root).
          Immediate -- but only fires near expression end, not on every
          closing paren in nested expressions.
        """
        candidate = self.phase  # default: no change

        if depth > self.prev_depth:
            # Opening new nesting level -- always attack (immediate)
            return "attack"

        if self.phase == "attack" and freedom < self.peak_freedom * 0.3:
            candidate = "decay"
        elif depth < self.prev_depth and depth <= 1:
            # Closing toward root -- release (immediate, depth-gated)
            return "release"
        elif self.phase == "decay" and self.phase_position > 2:
            # Stable after decay -- sustain (existing time gate)
            return "sustain"

        # Hysteresis for non-immediate transitions (currently: decay)
        if candidate != self.phase:
            if candidate == self.pending_phase:
                self.pending_count += 1
                if self.pending_count >= 2:
                    # Confirmed: transition
                    self.pending_phase = None
                    self.pending_count = 0
                    return candidate
            else:
                # New candidate -- start counting
                self.pending_phase = candidate
                self.pending_count = 1
            return self.phase  # hold current phase until confirmed
        else:
            # No transition requested -- reset pending
            self.pending_phase = None
            self.pending_count = 0
            return self.phase

    def advance_phase(self, freedom: float, depth: int) -> None:
        """Update phase state after a token."""
        new_phase = self.detect_phase(freedom, depth)
        if new_phase != self.phase:
            # Attack and release are immediate — bypass min_phase_duration.
            # Only non-immediate transitions (decay) are subject to it.
            immediate = new_phase in ("attack", "release")
            if not immediate and self.phase_position < self.min_phase_duration:
                # Too soon -- hold current phase
                pass
            else:
                self.phase = new_phase
                self.phase_position = 0
                if new_phase == "attack":
                    self.peak_freedom = freedom
        else:
            self.phase_position += 1
        # Track peak freedom during attack
        if self.phase == "attack":
            self.peak_freedom = max(self.peak_freedom, freedom)
        self.prev_depth = depth
        self.prev_freedom = freedom


@dataclass(frozen=True)
class EnvelopeTelemetry:
    """Per-token envelope telemetry record."""

    phase: str
    phase_position: int
    depth: int
    source_vector: list[float]  # 11 conditioned values
    modulation_vector: list[float]  # 7 output values
    final_temperature: float
    final_epsilon: float


def condition_source(
    raw: float,
    cfg: SourceConditionerConfig,
    prev_smoothed: float,
) -> float:
    """Normalize, rectify, and slew-limit a single source value."""
    normalized = (raw - cfg.range_min) / max(cfg.range_max - cfg.range_min, 1e-10)
    normalized = max(0.0, min(1.0, normalized))
    if cfg.invert:
        normalized = 1.0 - normalized
    if cfg.rectify:
        normalized = max(0.0, normalized)
    return cfg.slew_rate * normalized + (1.0 - cfg.slew_rate) * prev_smoothed


# fmt: off
DEFAULT_MATRIX: list[list[float]] = [
    # temp   top_p  rep    eps    open   back*  pres
    # *back (col 5) reserved -- zeroed, not wired in v1
    #
    # Temperature regime: base=0.05, attack peaks at ~0.15,
    # sustain drops to ~0.01. Matches old GrammarTemperatureHook's
    # near-deterministic behavior during constrained generation.
    # The ADSR shape modulates within a narrow band — the model
    # needs precision for argument values, not exploration.
    [ 0.5,   0.2,   0.0,   0.0,   0.0,   0.0,   0.0],  # 0: freedom (scaled by freedom ~0.00005)
    [ 0.01,  0.1,   0.0,   0.0,   0.0,   0.0,   0.0],  # 1: entropy
    [-0.02,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 2: confidence (high uncertainty → raise temp)
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,   0.0,   0.0],  # 3: overlap
    [ 0.0,   0.0,   0.0,   0.0, -80.0,   0.0,   0.0],  # 4: depth (stronger opener penalty)
    [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 5: position
    [ 0.10,  0.05, 10.0,  -0.05,  0.0,   0.0,   0.0],  # 6: attack (+0.10 → temp ~0.15)
    [ 0.02,  0.0,   5.0,   0.0,   0.0,   0.0,   0.0],  # 7: decay
    [-0.04, -0.1, -10.0,   0.1,   0.0,   0.0,   0.0],  # 8: sustain (-0.04 → temp ~0.01)
    [-0.02,  0.1,  10.0,  -0.05, -80.0,  0.0,   0.0],  # 9: release
    [-0.05,  0.0, -30.0,   0.0,   0.0,   0.0,   0.0],  # 10: cycle (drops temp too)
]
# fmt: on

DEFAULT_MATRIX_FLAT: tuple[float, ...] = tuple(
    v for row in DEFAULT_MATRIX for v in row
)

DEFAULT_CONDITIONERS: tuple[SourceConditionerConfig, ...] = (
    SourceConditionerConfig(range_min=0.0, range_max=1.0),   # 0: freedom
    SourceConditionerConfig(range_min=0.0, range_max=12.5),  # 1: entropy
    SourceConditionerConfig(                                  # 2: confidence
        range_min=-5.0, range_max=0.0, invert=True,
    ),
    SourceConditionerConfig(range_min=0.0, range_max=1.0),   # 3: overlap
    SourceConditionerConfig(range_min=0.0, range_max=10.0),  # 4: depth
    SourceConditionerConfig(range_min=0.0, range_max=1.0),   # 5: position
    SourceConditionerConfig(),  # 6: phase_attack (0 or 1)
    SourceConditionerConfig(),  # 7: phase_decay
    SourceConditionerConfig(),  # 8: phase_sustain
    SourceConditionerConfig(),  # 9: phase_release
    SourceConditionerConfig(),  # 10: cycle_detected
)


@dataclass(frozen=True)
class EnvelopeConfig:
    """Complete modulation matrix configuration."""

    # Base values (destinations start here, modulations are added)
    base_temperature: float = 0.05
    base_top_p: float = 0.9
    base_repetition_bias: float = -20.0
    base_epsilon: float = 0.1
    base_opener_bias: float = 0.0
    # reserved: col 5 zeroed, not wired in v1
    base_backtrack_threshold: float = -0.5
    base_presence_penalty: float = 0.0

    # Output clamp ranges
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.1, 1.0)
    repetition_bias_range: tuple[float, float] = (-100.0, 0.0)
    epsilon_range: tuple[float, float] = (0.01, 1.0)

    # Source conditioners (11 entries)
    conditioners: tuple[SourceConditionerConfig, ...] = field(
        default_factory=lambda: DEFAULT_CONDITIONERS,
    )

    # The matrix itself -- shape (11, 7), stored as flat tuple
    matrix_flat: tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_MATRIX_FLAT,
    )

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (11, 7)


class ModMatrixHookMlx:
    """Phase-aware modulation matrix for constrained generation.

    Routes 11 source signals through an (11, 7) matrix to produce
    7 parameter modulations per token. Replaces GrammarTemperatureHook
    and NestingDepthHook with a single configurable hook.
    """

    def __init__(
        self,
        config: EnvelopeConfig,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
        max_tokens: int = 128,
    ) -> None:
        import mlx.core as mx

        self.config = config
        self.max_tokens = max_tokens
        self._state = EnvelopeState(prev_smoothed=[0.0] * 11)
        self.last_telemetry: EnvelopeTelemetry | None = None

        # Build mod matrix as mx.array from flat config
        rows, cols = config.matrix_shape
        self._mod_matrix = mx.array(
            list(config.matrix_flat)
        ).reshape(rows, cols)

        # Pre-compute per-token paren delta and classify openers
        # (reuses NestingDepthHookMlx pattern)
        self._delta: dict[int, int] = {}
        self._opener_ids: set[int] = set()
        for tid in range(vocab_size):
            text = tokenizer_decode([tid])
            opens = text.count("(")
            closes = text.count(")")
            delta = opens - closes
            if delta != 0:
                self._delta[tid] = delta
            if opens > closes:
                self._opener_ids.add(tid)

    def reset(self) -> None:
        """Reset state for new constrained generation pass."""
        self._state = EnvelopeState(prev_smoothed=[0.0] * 11)
        self.last_telemetry = None

    def advance(self, token_id: int) -> None:
        """Update depth after token sampled."""
        self._state.depth += self._delta.get(token_id, 0)
        self._state.depth = max(self._state.depth, 0)

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        """Compute modulated parameters via matrix multiply."""
        import mlx.core as mx

        from tgirl.sample import detect_cycle

        vocab_size = logits.shape[0]
        cfg = self.config

        # 1. Compute raw source signals (MLX ops, scalar extract)
        freedom = float(mx.sum(valid_mask).item()) / vocab_size
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(mx.clip(probs, 1e-30, None))
        entropy = -float(mx.sum(probs * log_probs).item())
        overlap = float(
            mx.sum(probs * valid_mask.astype(mx.float32)).item()
        )
        confidence = float(mx.max(log_probs).item())

        # 2. Detect phase
        self._state.advance_phase(freedom, self._state.depth)
        phase = self._state.phase

        # 3. Cycle detection
        cycle = (
            1.0
            if detect_cycle(token_history) is not None
            else 0.0
        )

        # 4. Build source vector (conditioned)
        # n=11 Python scalars — documented deviation, sub-microsecond
        raw_sources = [
            freedom,
            entropy,
            confidence,
            overlap,
            float(self._state.depth),
            position / self.max_tokens,
            1.0 if phase == "attack" else 0.0,
            1.0 if phase == "decay" else 0.0,
            1.0 if phase == "sustain" else 0.0,
            1.0 if phase == "release" else 0.0,
            cycle,
        ]
        conditioned = []
        for i, (raw, scfg) in enumerate(
            zip(raw_sources, cfg.conditioners, strict=True)
        ):
            val = condition_source(
                raw, scfg, self._state.prev_smoothed[i]
            )
            self._state.prev_smoothed[i] = val
            conditioned.append(val)

        # 5. Matrix multiply (native MLX)
        source_vec = mx.array(conditioned)
        modulations = source_vec @ self._mod_matrix  # shape (7,)

        # 6. Apply base values + clamp (7 scalar .item() calls)
        temperature = max(
            cfg.temperature_range[0],
            min(
                cfg.temperature_range[1],
                cfg.base_temperature + float(modulations[0].item()),
            ),
        )
        top_p = max(
            cfg.top_p_range[0],
            min(
                cfg.top_p_range[1],
                cfg.base_top_p + float(modulations[1].item()),
            ),
        )
        rep_bias = max(
            cfg.repetition_bias_range[0],
            min(
                cfg.repetition_bias_range[1],
                cfg.base_repetition_bias
                + float(modulations[2].item()),
            ),
        )
        epsilon = max(
            cfg.epsilon_range[0],
            min(
                cfg.epsilon_range[1],
                cfg.base_epsilon + float(modulations[3].item()),
            ),
        )
        opener_bias_val = cfg.base_opener_bias + float(
            modulations[4].item()
        )
        # col 5 (backtrack) reserved — not read in v1
        presence = cfg.base_presence_penalty + float(
            modulations[6].item()
        )

        # 7. Build logit_bias for opener penalty
        logit_bias: dict[int, float] | None = None
        if opener_bias_val < -1.0:
            logit_bias = {
                tid: opener_bias_val
                for tid in self._opener_ids
            }

        # 8. Repetition bias via logit_bias on repeated tokens
        if rep_bias < cfg.base_repetition_bias:
            recent = token_history[-8:]
            counts: dict[int, int] = {}
            for tid in recent:
                counts[tid] = counts.get(tid, 0) + 1
            for tid, count in counts.items():
                if count > 2:
                    bias = rep_bias * (count - 2)
                    if logit_bias is None:
                        logit_bias = {}
                    existing = logit_bias.get(tid, 0.0)
                    logit_bias[tid] = max(
                        existing + bias, -100.0
                    )

        # 9. Record telemetry
        mod_vec = [
            float(modulations[i].item()) for i in range(7)
        ]
        self.last_telemetry = EnvelopeTelemetry(
            phase=phase,
            phase_position=self._state.phase_position,
            depth=self._state.depth,
            source_vector=list(conditioned),
            modulation_vector=mod_vec,
            final_temperature=temperature,
            final_epsilon=epsilon,
        )

        return ModelIntervention(
            temperature=temperature,
            top_p=top_p,
            logit_bias=logit_bias if logit_bias else None,
            transport_epsilon=epsilon,
            presence_penalty=(
                presence if presence != 0.0 else None
            ),
        )


class ModMatrixHook:
    """Phase-aware modulation matrix for constrained generation (torch).

    Torch variant of ModMatrixHookMlx. Same EnvelopeConfig and
    EnvelopeState (pure Python), different tensor ops in pre_forward.
    Conforms to InferenceHook protocol (receives GrammarState).
    """

    def __init__(
        self,
        config: EnvelopeConfig,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
        max_tokens: int = 128,
    ) -> None:
        import torch as _torch

        self.config = config
        self.max_tokens = max_tokens
        self._state = EnvelopeState(prev_smoothed=[0.0] * 11)

        # Build mod matrix as torch.Tensor
        rows, cols = config.matrix_shape
        self._mod_matrix = _torch.tensor(
            list(config.matrix_flat),
            dtype=_torch.float32,
        ).reshape(rows, cols)

        # Pre-compute per-token paren delta and classify openers
        self._delta: dict[int, int] = {}
        self._opener_ids: set[int] = set()
        for tid in range(vocab_size):
            text = tokenizer_decode([tid])
            opens = text.count("(")
            closes = text.count(")")
            delta = opens - closes
            if delta != 0:
                self._delta[tid] = delta
            if opens > closes:
                self._opener_ids.add(tid)

    def reset(self) -> None:
        """Reset state for new constrained generation pass."""
        self._state = EnvelopeState(prev_smoothed=[0.0] * 11)

    def advance(self, token_id: int) -> None:
        """Update depth after token sampled."""
        self._state.depth += self._delta.get(token_id, 0)
        self._state.depth = max(self._state.depth, 0)

    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention:
        """Compute modulated parameters via torch matmul."""
        import torch as _torch

        from tgirl.sample import detect_cycle

        vocab_size = logits.shape[0]
        cfg = self.config

        # Get valid mask from grammar state
        valid_mask = grammar_state.get_valid_mask(vocab_size)

        # 1. Source signals (torch ops, scalar extract)
        freedom = (
            float(valid_mask.sum().item()) / vocab_size
        )
        probs = _torch.softmax(logits, dim=-1)
        log_probs = _torch.log(
            _torch.clamp(probs, min=1e-30)
        )
        entropy = -float(
            _torch.sum(probs * log_probs).item()
        )
        overlap = float(
            _torch.sum(
                probs * valid_mask.float()
            ).item()
        )
        confidence = float(
            _torch.max(log_probs).item()
        )

        # 2. Detect phase
        self._state.advance_phase(freedom, self._state.depth)
        phase = self._state.phase

        # 3. Cycle detection
        cycle = (
            1.0
            if detect_cycle(token_history) is not None
            else 0.0
        )

        # 4. Build conditioned source vector
        raw_sources = [
            freedom,
            entropy,
            confidence,
            overlap,
            float(self._state.depth),
            position / self.max_tokens,
            1.0 if phase == "attack" else 0.0,
            1.0 if phase == "decay" else 0.0,
            1.0 if phase == "sustain" else 0.0,
            1.0 if phase == "release" else 0.0,
            cycle,
        ]
        conditioned = []
        for i, (raw, scfg) in enumerate(
            zip(raw_sources, cfg.conditioners, strict=True)
        ):
            val = condition_source(
                raw, scfg, self._state.prev_smoothed[i]
            )
            self._state.prev_smoothed[i] = val
            conditioned.append(val)

        # 5. Matrix multiply (native torch)
        source_vec = _torch.tensor(
            conditioned, dtype=_torch.float32
        )
        modulations = source_vec @ self._mod_matrix

        # 6. Base values + clamp
        temperature = max(
            cfg.temperature_range[0],
            min(
                cfg.temperature_range[1],
                cfg.base_temperature
                + float(modulations[0].item()),
            ),
        )
        top_p = max(
            cfg.top_p_range[0],
            min(
                cfg.top_p_range[1],
                cfg.base_top_p
                + float(modulations[1].item()),
            ),
        )
        rep_bias = max(
            cfg.repetition_bias_range[0],
            min(
                cfg.repetition_bias_range[1],
                cfg.base_repetition_bias
                + float(modulations[2].item()),
            ),
        )
        epsilon = max(
            cfg.epsilon_range[0],
            min(
                cfg.epsilon_range[1],
                cfg.base_epsilon
                + float(modulations[3].item()),
            ),
        )
        opener_bias_val = cfg.base_opener_bias + float(
            modulations[4].item()
        )
        presence = cfg.base_presence_penalty + float(
            modulations[6].item()
        )

        # 7. Opener penalty
        logit_bias: dict[int, float] | None = None
        if opener_bias_val < -1.0:
            logit_bias = {
                tid: opener_bias_val
                for tid in self._opener_ids
            }

        # 8. Repetition bias
        if rep_bias < cfg.base_repetition_bias:
            recent = token_history[-8:]
            counts: dict[int, int] = {}
            for tid in recent:
                counts[tid] = counts.get(tid, 0) + 1
            for tid, count in counts.items():
                if count > 2:
                    bias = rep_bias * (count - 2)
                    if logit_bias is None:
                        logit_bias = {}
                    existing = logit_bias.get(tid, 0.0)
                    logit_bias[tid] = max(
                        existing + bias, -100.0
                    )

        return ModelIntervention(
            temperature=temperature,
            top_p=top_p,
            logit_bias=logit_bias if logit_bias else None,
            transport_epsilon=epsilon,
            presence_penalty=(
                presence if presence != 0.0 else None
            ),
        )
