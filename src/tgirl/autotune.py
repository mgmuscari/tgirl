"""Rule-based steering controller for ESTRADIOL-v2.

Reads the per-turn coherence + certainty telemetry, classifies the
output regime, and emits a control action over (α, β, temperature).

The thresholds and per-regime responses are calibrated against the
2D α × β sweep on Qwen3.5-0.8B. Three failure modes were observed:

- **sycophant_trap** (α≈0.4 with single-layer): short, helpful-template
  output, ``finish_reason=stop`` at <100 tokens, ``token_entropy>0.85``.
  Discrete attractor — escape by jittering α away from the basin.
- **loop_emergence** (α≈0.7): repeats climbing to ~0.16, entropy
  softening to ~0.55. Gentle α reduction restores signal.
- **loop_collapse** (α=0.9 with β=1.0): catastrophic — repeat_rate→0.83,
  entropy→0.08. Aggressive α reduction + β floored at 2.0 (where β=1.0
  at high α was the worst point in the entire grid).

The controller interface ``autotune(obs) → Action`` is what a learned
perceptron will eventually replace. Phase-1 rules live here so the
serve-layer integration is fully testable; the (obs, action) tuples
each call produces also serve as the training set for the perceptron
swap.

Hard policy constraints derived from the sweep:

- ``β ≥ 2.0`` floor when active. ``β = 1.0`` at high α is worse than
  no band at all. The controller never drives β below this.
- ``α ∈ [0.0, 1.0]`` clipped. Above 1.0 the residual-relative
  correction over-writes the signal entirely.
- ``temperature ∈ [0.0, 1.0]`` clipped. Raised only on collapse to
  add stochastic escape pressure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Regime = Literal["signal", "sycophant_trap", "loop_emergence", "loop_collapse"]


@dataclass(frozen=True)
class Observables:
    """The per-turn telemetry the controller reads.

    Mirrors what /v1/steering/status surfaces after each completion:
    the coherence triple, finish disposition, mean certainty triple,
    and current steering config.
    """

    # Post-generation coherence (from tgirl.coherence.compute_coherence)
    repeat_rate: float
    bigram_novelty: float
    token_entropy: float
    n_tokens: int
    finish_reason: str  # "stop" | "length"
    # Pre-sampling certainty (from tgirl.certainty.mean_certainty)
    mean_entropy: float
    mean_top1_prob: float
    mean_top1_margin: float
    # Current steering state
    alpha: float
    beta: float | None
    temperature: float


@dataclass(frozen=True)
class Action:
    """The controller's output: next-turn config + a label trail."""

    next_alpha: float
    next_beta: float | None
    next_temperature: float
    regime: Regime
    rationale: str


# Calibration constants. Sourced from the 2D sweep on Qwen3.5-0.8B-MLX-4bit.
# These are the "weights" of the rule-based controller; a learned perceptron
# would replace this with a parameter vector trained on (obs, action) tuples.
_BETA_FLOOR: float = 2.0  # β=1.0 at high α is catastrophic
_ALPHA_MAX: float = 1.0
_ALPHA_MIN: float = 0.0
_TEMP_MAX: float = 1.0
_TEMP_MIN: float = 0.0

# Regime thresholds
_COLLAPSE_REPEAT: float = 0.30  # α=0.9 region
_COLLAPSE_ENTROPY: float = 0.30
_EMERGENCE_REPEAT: float = 0.05
_EMERGENCE_ENTROPY: float = 0.55
_SYCOPHANT_N_TOKENS: int = 100
_SYCOPHANT_ENTROPY: float = 0.85
_SYCOPHANT_BASIN_CENTER: float = 0.45  # midpoint of α=0.3-0.5 trap

# Per-regime step sizes
_COLLAPSE_ALPHA_STEP: float = 0.20
_EMERGENCE_ALPHA_STEP: float = 0.10
_SYCOPHANT_ALPHA_STEP: float = 0.10
_COLLAPSE_TEMP_BUMP: float = 0.10


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_beta(beta: float | None) -> float | None:
    """Enforce the β safety floor: None (single-layer) is fine; any
    finite value must be ≥ _BETA_FLOOR.
    """
    if beta is None:
        return None
    return max(beta, _BETA_FLOOR)


def classify(obs: Observables) -> Regime:
    """Coarse regime label from the coherence + length signature.

    Order matters: severe collapse precedes emergence, and sycophant
    is checked only when the coherence numbers look "fine" (high H,
    no repeats) — its signature is *length*, not entropy.
    """
    # Severe collapse: pure repetitive token soup.
    if (
        obs.repeat_rate > _COLLAPSE_REPEAT
        or obs.token_entropy < _COLLAPSE_ENTROPY
    ):
        return "loop_collapse"

    # Sycophant trap: short, EOS-stopped, high entropy. The α=0.4
    # discrete attractor — looks "coherent" by entropy but has
    # collapsed into the helpful-assistant template basin.
    if (
        obs.n_tokens < _SYCOPHANT_N_TOKENS
        and obs.finish_reason == "stop"
        and obs.token_entropy > _SYCOPHANT_ENTROPY
        and obs.repeat_rate < _EMERGENCE_REPEAT
    ):
        return "sycophant_trap"

    # Emerging loop: repeats climbing or entropy softening.
    if (
        obs.repeat_rate >= _EMERGENCE_REPEAT
        or obs.token_entropy < _EMERGENCE_ENTROPY
    ):
        return "loop_emergence"

    return "signal"


def autotune(obs: Observables) -> Action:
    """Map an observation to the next-turn (α, β, temperature) config.

    Pure function: no I/O, no state. The serve layer threads the
    Action's next_* fields back into the next request and logs
    (obs, action) for the training set.
    """
    regime = classify(obs)
    next_alpha = obs.alpha
    next_beta = obs.beta
    next_temp = obs.temperature
    rationale = ""

    if regime == "signal":
        rationale = "in target regime; holding parameters"

    elif regime == "loop_collapse":
        # Aggressive α reduction. Force β to safe floor (or clear it).
        # Bump temperature slightly to add stochastic escape pressure.
        next_alpha = _clip(
            obs.alpha - _COLLAPSE_ALPHA_STEP, _ALPHA_MIN, _ALPHA_MAX
        )
        next_beta = _safe_beta(obs.beta)
        next_temp = _clip(
            obs.temperature + _COLLAPSE_TEMP_BUMP, _TEMP_MIN, _TEMP_MAX
        )
        rationale = (
            f"loop collapse: rep={obs.repeat_rate:.2f}, "
            f"H={obs.token_entropy:.2f}; α reduced, β floored, +temp"
        )

    elif regime == "loop_emergence":
        # Gentle α reduction; ensure β is in the safe band (2.0
        # consistently helps across the α range per the 2D sweep).
        next_alpha = _clip(
            obs.alpha - _EMERGENCE_ALPHA_STEP, _ALPHA_MIN, _ALPHA_MAX
        )
        next_beta = _safe_beta(obs.beta)
        rationale = (
            f"loop emerging: rep={obs.repeat_rate:.2f}, "
            f"H={obs.token_entropy:.2f}; α eased back"
        )

    else:  # sycophant_trap
        # Discrete attractor at α≈0.4. Push α away from the basin
        # center: above center → step down toward 0.3 escape; below
        # center → step up toward 0.5 escape. Both directions worked
        # in the 2D sweep.
        if obs.alpha > _SYCOPHANT_BASIN_CENTER:
            next_alpha = _clip(
                obs.alpha - _SYCOPHANT_ALPHA_STEP,
                _ALPHA_MIN, _ALPHA_MAX,
            )
            direction = "down"
        else:
            next_alpha = _clip(
                obs.alpha + _SYCOPHANT_ALPHA_STEP,
                _ALPHA_MIN, _ALPHA_MAX,
            )
            direction = "up"
        # Switching from single-layer to β=2.0 is also known to break
        # out (the 2D sweep showed β=2.0 lengthens output at α=0.3,
        # 0.5). If we're not already in the safe band, jump there.
        next_beta = _BETA_FLOOR if obs.beta is None else _safe_beta(obs.beta)
        rationale = (
            f"sycophant trap: n_tok={obs.n_tokens}, H={obs.token_entropy:.2f}; "
            f"α jittered {direction}, β→{next_beta}"
        )

    return Action(
        next_alpha=next_alpha,
        next_beta=next_beta,
        next_temperature=next_temp,
        regime=regime,
        rationale=rationale,
    )


def observables_to_dict(obs: Observables) -> dict[str, Any]:
    """JSON-friendly form for the training-set logger."""
    return asdict(obs)


def action_to_dict(action: Action) -> dict[str, Any]:
    """JSON-friendly form for the training-set logger."""
    return asdict(action)
