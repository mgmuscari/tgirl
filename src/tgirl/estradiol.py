"""ESTRADIOL v2: Closed-loop behavioral steering via low-rank codebook.

Provides the controller, data structures, and calibration I/O for
activation-level behavioral steering at the transformer bottleneck layer.

This module has ZERO coupling to any other tgirl module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SteeringState:
    """Per-forward-pass steering configuration.

    Passed into the steerable forward function to configure
    probe (read) and injection (write) at the bottleneck layer.
    """

    V_basis: Any  # mx.array (d_model, K) — codebook basis
    delta_alpha: Any  # mx.array (K,) — correction to inject
    bottleneck_layer: int


@dataclass
class CalibrationResult:
    """Output of the calibration pipeline. Everything needed to steer a model.

    Serialized to/from .estradiol files (safetensors format).
    """

    V_basis: Any  # mx.array (d_model, K) float16
    bottleneck_layer: int
    K: int
    model_id: str
    trait_map: dict[str, Any]  # {name: mx.array (K,)}
    scaffold_basis: Any  # mx.array (d_model, r_scaffold) float16
    scaffold_rank: int
    timestamp: str
    eff_rank: float
    compression_ratio: float
    n_dims: int
    complement_fractions: dict[str, float]  # {name: fraction in complement}


@runtime_checkable
class EstradiolControllerProto(Protocol):
    """Public surface of an ESTRADIOL controller.

    Documents the duck-typed contract that sample_mlx.py relies on.
    Concrete implementation: ``EstradiolController`` (below).
    """

    V_basis: Any  # mx.array (d_model, K)
    alpha_current: Any  # mx.array (K,)

    def step(self, probe_alpha: Any) -> Any: ...

    def make_steering_state(self, delta_alpha: Any) -> SteeringState: ...

    def reset(self) -> None: ...


class EstradiolController:
    """Closed-loop proportional controller with EMA smoothing.

    Reads probe_alpha (K-dim behavioral state) each token,
    updates an EMA of the behavioral state, and returns a
    correction delta_alpha toward the target.

    Cost: ~55 FLOPs per step (scalar arithmetic on K-dim vectors).
    """

    def __init__(
        self,
        V_basis: Any,  # mx.array (d_model, K)
        bottleneck_layer: int,
        alpha_target: Any,  # mx.array (K,)
        gain: float = 0.1,
        ema_beta: float = 0.9,
    ) -> None:
        import mlx.core as mx

        self.V_basis = V_basis
        self.bottleneck_layer = bottleneck_layer
        self.alpha_target = alpha_target
        self.gain = gain
        self.ema_beta = ema_beta
        self.alpha_current = mx.zeros_like(alpha_target)

    def step(self, probe_alpha: Any) -> Any:
        """Probe reading in, correction delta out."""
        self.alpha_current = (
            self.ema_beta * self.alpha_current
            + (1.0 - self.ema_beta) * probe_alpha
        )
        return self.gain * (self.alpha_target - self.alpha_current)

    def make_steering_state(self, delta_alpha: Any) -> SteeringState:
        return SteeringState(
            V_basis=self.V_basis,
            delta_alpha=delta_alpha,
            bottleneck_layer=self.bottleneck_layer,
        )

    def reset(self) -> None:
        import mlx.core as mx

        self.alpha_current = mx.zeros_like(self.alpha_target)


def save_estradiol(path: Any, result: CalibrationResult) -> None:
    """Serialize a CalibrationResult to a single .estradiol file (safetensors)."""
    import json
    from pathlib import Path

    import numpy as np
    from safetensors.numpy import save_file

    path = Path(path)

    tensors: dict[str, np.ndarray] = {
        "V_basis": np.array(result.V_basis.astype(_mx().float16)),
        "scaffold_basis": np.array(result.scaffold_basis.astype(_mx().float16)),
    }
    for name, vec in result.trait_map.items():
        tensors[f"trait/{name}"] = np.array(vec.astype(_mx().float32))

    metadata = {
        "model_id": result.model_id,
        "bottleneck_layer": str(result.bottleneck_layer),
        "K": str(result.K),
        "scaffold_rank": str(result.scaffold_rank),
        "timestamp": result.timestamp,
        "eff_rank": str(result.eff_rank),
        "compression_ratio": str(result.compression_ratio),
        "n_dims": str(result.n_dims),
        "complement_fractions": json.dumps(result.complement_fractions),
        "trait_names": json.dumps(sorted(result.trait_map.keys())),
    }

    save_file(tensors, str(path), metadata=metadata)


def load_estradiol(path: Any) -> CalibrationResult:
    """Load a CalibrationResult from a .estradiol file (safetensors)."""
    import json
    from pathlib import Path

    from safetensors.numpy import load_file

    path = Path(path)
    if not path.exists():
        msg = f"No such file: {path}"
        raise FileNotFoundError(msg)

    mx = _mx()

    tensors = load_file(str(path))
    # safetensors metadata requires re-reading the header
    from typing import cast

    from safetensors import safe_open

    with cast(Any, safe_open)(str(path), framework="numpy") as f:
        meta = f.metadata()

    V_basis = mx.array(tensors["V_basis"]).astype(mx.float32)
    scaffold_basis = mx.array(tensors["scaffold_basis"]).astype(mx.float32)

    trait_names = json.loads(meta["trait_names"])
    trait_map = {
        name: mx.array(tensors[f"trait/{name}"]) for name in trait_names
    }
    complement_fractions = json.loads(meta["complement_fractions"])

    return CalibrationResult(
        V_basis=V_basis,
        bottleneck_layer=int(meta["bottleneck_layer"]),
        K=int(meta["K"]),
        model_id=meta["model_id"],
        trait_map=trait_map,
        scaffold_basis=scaffold_basis,
        scaffold_rank=int(meta["scaffold_rank"]),
        timestamp=meta["timestamp"],
        eff_rank=float(meta["eff_rank"]),
        compression_ratio=float(meta["compression_ratio"]),
        n_dims=int(meta["n_dims"]),
        complement_fractions=complement_fractions,
    )


def _mx() -> Any:
    """Lazy import mlx.core."""
    import mlx.core as mx

    return mx
