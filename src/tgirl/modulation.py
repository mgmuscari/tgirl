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
