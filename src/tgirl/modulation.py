"""ADSR modulation matrix for constrained generation.

Routes source signals through a configurable (11, 7) matrix to produce
per-token sampling parameter modulations. Replaces GrammarTemperatureHook
and NestingDepthHook with a single unified hook.

Design: docs/design/adsr-envelope.md
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    [ 0.3,   0.2,   0.0,   0.0,   0.0,   0.0,   0.0],  # 0: freedom
    [ 0.1,   0.1,   0.0,   0.0,   0.0,   0.0,   0.0],  # 1: entropy
    [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 2: confidence
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,   0.0,   0.0],  # 3: overlap
    [ 0.0,   0.0,   0.0,   0.0, -20.0,   0.0,   0.0],  # 4: depth
    [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # 5: position
    [ 0.3,   0.35, 10.0,  -0.05,  0.0,   0.0,   0.0],  # 6: attack
    [ 0.0,   0.2,   5.0,   0.0,   0.0,   0.0,   0.0],  # 7: decay
    [-0.15, -0.1, -10.0,   0.1,   0.0,   0.0,   0.0],  # 8: sustain
    [-0.05,  0.3,  10.0,  -0.05, -50.0,  0.0,   0.0],  # 9: release
    [ 0.0,   0.0, -30.0,   0.0,   0.0,   0.0,   0.0],  # 10: cycle
]
# fmt: on

DEFAULT_MATRIX_FLAT: tuple[float, ...] = tuple(
    v for row in DEFAULT_MATRIX for v in row
)


@dataclass(frozen=True)
class EnvelopeConfig:
    """Complete modulation matrix configuration."""

    # Base values (destinations start here, modulations are added)
    base_temperature: float = 0.3
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
