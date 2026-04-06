"""Tests for tgirl.estradiol — behavioral steering data structures and controller."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest


class TestSteeringState:
    """SteeringState is a frozen container for per-forward-pass steering config."""

    def test_fields(self) -> None:
        from tgirl.estradiol import SteeringState

        V = mx.ones((1024, 11))
        delta = mx.zeros((11,))
        ss = SteeringState(V_basis=V, delta_alpha=delta, bottleneck_layer=14)
        assert ss.V_basis.shape == (1024, 11)
        assert ss.delta_alpha.shape == (11,)
        assert ss.bottleneck_layer == 14

    def test_frozen(self) -> None:
        from tgirl.estradiol import SteeringState

        ss = SteeringState(
            V_basis=mx.ones((1024, 11)),
            delta_alpha=mx.zeros((11,)),
            bottleneck_layer=14,
        )
        with pytest.raises((AttributeError, TypeError)):
            ss.bottleneck_layer = 5  # type: ignore[misc]


class TestCalibrationResult:
    """CalibrationResult stores everything needed to instantiate a controller."""

    def test_fields(self) -> None:
        from tgirl.estradiol import CalibrationResult

        K = 11
        d_model = 1024
        V = mx.ones((d_model, K))
        scaffold = mx.ones((d_model, 6))
        trait_map = {"helpful": mx.ones((K,)), "honest": mx.ones((K,))}

        cr = CalibrationResult(
            V_basis=V,
            bottleneck_layer=14,
            K=K,
            model_id="Qwen/Qwen3.5-0.8B",
            trait_map=trait_map,
            scaffold_basis=scaffold,
            scaffold_rank=6,
            timestamp="2026-04-05T00:00:00Z",
            eff_rank=10.5,
            compression_ratio=0.58,
            n_dims=25,
            complement_fractions={"helpful": 0.85, "honest": 0.66},
        )
        assert cr.K == K
        assert cr.bottleneck_layer == 14
        assert cr.V_basis.shape == (d_model, K)
        assert cr.scaffold_basis.shape == (d_model, 6)
        assert len(cr.trait_map) == 2
        assert cr.trait_map["helpful"].shape == (K,)
        assert cr.complement_fractions["helpful"] == pytest.approx(0.85)


class TestForwardResult:
    """ForwardResult lives in cache.py — logits + optional probe reading."""

    def test_logits_and_probe(self) -> None:
        from tgirl.cache import ForwardResult

        logits = mx.ones((32000,))
        probe = mx.ones((11,))
        fr = ForwardResult(logits=logits, probe_alpha=probe)
        assert fr.logits.shape == (32000,)
        assert fr.probe_alpha.shape == (11,)

    def test_logits_only(self) -> None:
        from tgirl.cache import ForwardResult

        logits = mx.ones((32000,))
        fr = ForwardResult(logits=logits, probe_alpha=None)
        assert fr.logits.shape == (32000,)
        assert fr.probe_alpha is None


class TestEstradiolController:
    """Proportional controller with EMA smoothing."""

    @pytest.fixture()
    def controller(self):
        from tgirl.estradiol import EstradiolController

        K = 4
        d_model = 16
        V = mx.random.normal((d_model, K))
        alpha_target = mx.array([1.0, 0.0, -1.0, 0.5])
        return EstradiolController(
            V_basis=V,
            bottleneck_layer=5,
            alpha_target=alpha_target,
            gain=0.1,
            ema_beta=0.9,
        )

    def test_step_returns_delta_shape(self, controller) -> None:
        probe = mx.zeros((4,))
        delta = controller.step(probe)
        mx.eval(delta)
        assert delta.shape == (4,)

    def test_zero_probe_nonzero_target_gives_positive_delta(self, controller) -> None:
        """When current state is zero and target is nonzero, correction pushes toward target."""
        probe = mx.zeros((4,))
        delta = controller.step(probe)
        mx.eval(delta)
        # delta = gain * (target - current), current=0.1*probe after one step
        # target = [1, 0, -1, 0.5], so delta[0] > 0, delta[2] < 0
        assert float(delta[0].item()) > 0
        assert float(delta[2].item()) < 0

    def test_steady_state_delta_is_zero(self, controller) -> None:
        """After many steps at the target, correction converges to zero."""
        target = controller.alpha_target
        for _ in range(200):
            delta = controller.step(target)
        mx.eval(delta)
        for i in range(4):
            assert abs(float(delta[i].item())) < 1e-4

    def test_gain_scaling(self) -> None:
        """Delta magnitude scales linearly with gain."""
        from tgirl.estradiol import EstradiolController

        K, d_model = 4, 16
        V = mx.random.normal((d_model, K))
        target = mx.array([1.0, 0.0, 0.0, 0.0])
        probe = mx.zeros((K,))

        c1 = EstradiolController(V, 5, target, gain=0.1, ema_beta=0.0)
        c2 = EstradiolController(V, 5, target, gain=0.5, ema_beta=0.0)
        d1 = c1.step(probe)
        d2 = c2.step(probe)
        mx.eval(d1, d2)

        ratio = float(mx.linalg.norm(d2).item()) / float(mx.linalg.norm(d1).item())
        assert ratio == pytest.approx(5.0, rel=0.01)

    def test_reset_zeros_state(self, controller) -> None:
        probe = mx.ones((4,))
        controller.step(probe)
        controller.reset()
        # After reset, alpha_current should be zero
        assert float(mx.linalg.norm(controller.alpha_current).item()) == 0.0

    def test_make_steering_state(self, controller) -> None:
        from tgirl.estradiol import SteeringState

        delta = mx.array([0.1, -0.2, 0.3, 0.0])
        ss = controller.make_steering_state(delta)
        assert isinstance(ss, SteeringState)
        assert ss.bottleneck_layer == 5
        assert ss.delta_alpha.shape == (4,)

    def test_ema_halflife(self) -> None:
        """At beta=0.9, half-life is ~6.6 tokens. After 7 steps with constant
        probe=1.0, alpha_current should be > 0.5."""
        from tgirl.estradiol import EstradiolController

        K, d_model = 1, 4
        V = mx.ones((d_model, K))
        c = EstradiolController(V, 0, mx.zeros((K,)), gain=0.0, ema_beta=0.9)
        probe = mx.ones((K,))
        for _ in range(7):
            c.step(probe)
        mx.eval(c.alpha_current)
        assert float(c.alpha_current[0].item()) > 0.5


class TestCalibrationIO:
    """Save/load .estradiol files (safetensors + metadata)."""

    @pytest.fixture()
    def cal_result(self):
        from tgirl.estradiol import CalibrationResult

        K, d_model = 11, 1024
        return CalibrationResult(
            V_basis=mx.random.normal((d_model, K)),
            bottleneck_layer=14,
            K=K,
            model_id="Qwen/Qwen3.5-0.8B",
            trait_map={
                "helpful": mx.random.normal((K,)),
                "honest": mx.random.normal((K,)),
            },
            scaffold_basis=mx.random.normal((d_model, 6)),
            scaffold_rank=6,
            timestamp="2026-04-05T00:00:00Z",
            eff_rank=10.5,
            compression_ratio=0.58,
            n_dims=25,
            complement_fractions={"helpful": 0.85, "honest": 0.66},
        )

    def test_roundtrip(self, cal_result, tmp_path: Path) -> None:
        from tgirl.estradiol import load_estradiol, save_estradiol

        path = tmp_path / "test.estradiol"
        save_estradiol(path, cal_result)
        loaded = load_estradiol(path)

        assert loaded.K == cal_result.K
        assert loaded.bottleneck_layer == cal_result.bottleneck_layer
        assert loaded.model_id == cal_result.model_id
        assert loaded.scaffold_rank == cal_result.scaffold_rank
        assert loaded.n_dims == cal_result.n_dims
        assert loaded.eff_rank == pytest.approx(cal_result.eff_rank, rel=1e-3)
        assert loaded.compression_ratio == pytest.approx(
            cal_result.compression_ratio, rel=1e-3
        )
        assert loaded.V_basis.shape == cal_result.V_basis.shape
        assert loaded.scaffold_basis.shape == cal_result.scaffold_basis.shape

    def test_trait_map_preserved(self, cal_result, tmp_path: Path) -> None:
        from tgirl.estradiol import load_estradiol, save_estradiol

        path = tmp_path / "test.estradiol"
        save_estradiol(path, cal_result)
        loaded = load_estradiol(path)

        assert set(loaded.trait_map.keys()) == {"helpful", "honest"}
        assert loaded.trait_map["helpful"].shape == (cal_result.K,)

    def test_complement_fractions_preserved(self, cal_result, tmp_path: Path) -> None:
        from tgirl.estradiol import load_estradiol, save_estradiol

        path = tmp_path / "test.estradiol"
        save_estradiol(path, cal_result)
        loaded = load_estradiol(path)

        assert loaded.complement_fractions["helpful"] == pytest.approx(0.85, rel=1e-3)
        assert loaded.complement_fractions["honest"] == pytest.approx(0.66, rel=1e-3)

    def test_float16_precision(self, cal_result, tmp_path: Path) -> None:
        """V_basis is stored as float16 — verify reasonable precision."""
        from tgirl.estradiol import load_estradiol, save_estradiol

        path = tmp_path / "test.estradiol"
        save_estradiol(path, cal_result)
        loaded = load_estradiol(path)

        # float16 has ~3 decimal digits of precision
        diff = mx.max(mx.abs(loaded.V_basis - cal_result.V_basis.astype(mx.float16).astype(mx.float32)))
        mx.eval(diff)
        assert float(diff.item()) < 0.01

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        from tgirl.estradiol import load_estradiol

        with pytest.raises(FileNotFoundError):
            load_estradiol(tmp_path / "nonexistent.estradiol")

    def test_file_is_single_file(self, cal_result, tmp_path: Path) -> None:
        """The .estradiol output should be a single file, not a directory."""
        from tgirl.estradiol import save_estradiol

        path = tmp_path / "test.estradiol"
        save_estradiol(path, cal_result)
        assert path.is_file()
