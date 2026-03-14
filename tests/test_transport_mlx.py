"""Tests for tgirl.transport_mlx — MLX-native optimal transport logit redistribution."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch


class TestTransportResultMlx:
    """TransportResultMlx is a NamedTuple with mx.array logits."""

    def test_result_is_namedtuple(self) -> None:
        from tgirl.transport_mlx import TransportResultMlx

        assert hasattr(TransportResultMlx, "_fields")
        assert "logits" in TransportResultMlx._fields
        assert "wasserstein_distance" in TransportResultMlx._fields

    def test_result_tuple_unpacking(self) -> None:
        from tgirl.transport_mlx import TransportResultMlx

        logits = mx.array([1.0, 2.0])
        r = TransportResultMlx(
            logits=logits,
            wasserstein_distance=0.5,
            bypassed=False,
            bypass_reason=None,
            iterations=10,
        )
        assert r[1] == 0.5
        assert r.bypassed is False

    def test_result_logits_is_mx_array(self) -> None:
        from tgirl.transport_mlx import TransportResultMlx

        logits = mx.array([1.0, 2.0])
        r = TransportResultMlx(
            logits=logits,
            wasserstein_distance=0.5,
            bypassed=False,
            bypass_reason=None,
            iterations=0,
        )
        assert isinstance(r.logits, mx.array)


class TestBypassMlx:
    """_check_bypass_mlx detects when OT should be skipped."""

    def test_bypass_forced_decode_zero_valid(self) -> None:
        from tgirl.transport import TransportConfig
        from tgirl.transport_mlx import _check_bypass_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        mask = mx.array([False, False, False])
        should, reason = _check_bypass_mlx(logits, mask, TransportConfig())
        assert should is True
        assert reason == "forced_decode"

    def test_bypass_forced_decode_one_valid(self) -> None:
        from tgirl.transport import TransportConfig
        from tgirl.transport_mlx import _check_bypass_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        mask = mx.array([False, True, False])
        should, reason = _check_bypass_mlx(logits, mask, TransportConfig())
        assert should is True
        assert reason == "forced_decode"

    def test_bypass_valid_ratio_high(self) -> None:
        from tgirl.transport import TransportConfig
        from tgirl.transport_mlx import _check_bypass_mlx

        logits = mx.array([1.0, 2.0, 3.0, 4.0])
        mask = mx.array([True, True, True, False])
        should, reason = _check_bypass_mlx(logits, mask, TransportConfig())
        assert should is True
        assert reason == "valid_ratio_high"

    def test_bypass_invalid_mass_low(self) -> None:
        from tgirl.transport import TransportConfig
        from tgirl.transport_mlx import _check_bypass_mlx

        logits = mx.array([10.0, 10.0, -100.0, -100.0, -100.0])
        mask = mx.array([True, True, False, False, False])
        should, reason = _check_bypass_mlx(logits, mask, TransportConfig())
        assert should is True
        assert reason == "invalid_mass_low"

    def test_no_bypass(self) -> None:
        from tgirl.transport import TransportConfig
        from tgirl.transport_mlx import _check_bypass_mlx

        logits = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mask = mx.array([True, True, False, False, False])
        should, reason = _check_bypass_mlx(logits, mask, TransportConfig())
        assert should is False
        assert reason is None


class TestStandardMaskingMlx:
    """_standard_masking_mlx sets invalid logits to -inf."""

    def test_invalid_become_neg_inf(self) -> None:
        from tgirl.transport_mlx import _standard_masking_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        mask = mx.array([True, False, True])
        result = _standard_masking_mlx(logits, mask)
        assert float(result[1]) == float("-inf")

    def test_valid_unchanged(self) -> None:
        from tgirl.transport_mlx import _standard_masking_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        mask = mx.array([True, False, True])
        result = _standard_masking_mlx(logits, mask)
        assert float(result[0]) == 1.0
        assert float(result[2]) == 3.0

    def test_output_shape(self) -> None:
        from tgirl.transport_mlx import _standard_masking_mlx

        logits = mx.array([1.0, 2.0, 3.0, 4.0])
        mask = mx.array([True, False, True, False])
        result = _standard_masking_mlx(logits, mask)
        assert result.shape == logits.shape


class TestSinkhornConvergenceMlx:
    """_sinkhorn_log_domain_mlx converges and produces valid transport plans."""

    def test_marginals_match(self) -> None:
        from tgirl.transport_mlx import _sinkhorn_log_domain_mlx

        cost = mx.array(np.random.rand(3, 4).astype(np.float32))
        source = mx.array([0.3, 0.3, 0.4])
        target = mx.array([0.2, 0.3, 0.2, 0.3])
        plan, _, _ = _sinkhorn_log_domain_mlx(
            cost, source, target, epsilon=0.1,
            max_iterations=100, convergence_threshold=1e-8,
        )
        row_sums = np.array(mx.sum(plan, axis=1))
        col_sums = np.array(mx.sum(plan, axis=0))
        np.testing.assert_allclose(row_sums, np.array(source), atol=1e-4)
        np.testing.assert_allclose(col_sums, np.array(target), atol=1e-4)

    def test_convergence_before_max_iter(self) -> None:
        from tgirl.transport_mlx import _sinkhorn_log_domain_mlx

        np.random.seed(42)
        cost = mx.array(np.random.rand(3, 4).astype(np.float32))
        source = mx.array([0.3, 0.3, 0.4])
        target = mx.array([0.2, 0.3, 0.2, 0.3])
        _, _, iters = _sinkhorn_log_domain_mlx(
            cost, source, target, epsilon=0.1,
            max_iterations=200, convergence_threshold=1e-6,
        )
        assert iters < 200

    def test_non_negative_wasserstein(self) -> None:
        from tgirl.transport_mlx import _sinkhorn_log_domain_mlx

        cost = mx.array(np.random.rand(3, 4).astype(np.float32))
        source = mx.array([0.3, 0.3, 0.4])
        target = mx.array([0.2, 0.3, 0.2, 0.3])
        _, w_dist, _ = _sinkhorn_log_domain_mlx(
            cost, source, target, epsilon=0.1,
            max_iterations=50, convergence_threshold=1e-6,
        )
        assert w_dist >= 0.0


class TestSinkhornMatchesTorch:
    """MLX and torch Sinkhorn produce plans within tolerance on identical inputs."""

    def test_plans_match(self) -> None:
        from tgirl.transport import _sinkhorn_log_domain
        from tgirl.transport_mlx import _sinkhorn_log_domain_mlx

        np.random.seed(42)
        cost_np = np.random.rand(3, 4).astype(np.float32)
        source_np = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        target_np = np.array([0.2, 0.3, 0.2, 0.3], dtype=np.float32)

        # Torch version
        plan_torch, w_torch, _ = _sinkhorn_log_domain(
            torch.from_numpy(cost_np),
            torch.from_numpy(source_np),
            torch.from_numpy(target_np),
            epsilon=0.1, max_iterations=100, convergence_threshold=1e-8,
        )

        # MLX version
        plan_mlx, w_mlx, _ = _sinkhorn_log_domain_mlx(
            mx.array(cost_np),
            mx.array(source_np),
            mx.array(target_np),
            epsilon=0.1, max_iterations=100, convergence_threshold=1e-8,
        )

        plan_mlx_np = np.array(plan_mlx)
        plan_torch_np = plan_torch.numpy()
        np.testing.assert_allclose(plan_mlx_np, plan_torch_np, atol=1e-4)
        assert abs(w_mlx - w_torch) < 1e-4


class TestRedistributeFullPathMlx:
    """End-to-end redistribution produces valid logits."""

    def test_full_path(self) -> None:
        from tgirl.transport_mlx import redistribute_logits_mlx

        np.random.seed(42)
        vocab_size = 10
        logits = mx.array(np.random.randn(vocab_size).astype(np.float32))
        valid_mask = mx.array([True, True, False, False, True,
                               False, False, True, False, False])
        embeddings = mx.array(np.random.randn(vocab_size, 16).astype(np.float32))

        result = redistribute_logits_mlx(logits, valid_mask, embeddings)
        assert isinstance(result.logits, mx.array)
        assert result.logits.shape == (vocab_size,)
        # Invalid positions should be -inf
        result_np = np.array(result.logits)
        mask_np = np.array(valid_mask)
        assert all(result_np[~mask_np] == float("-inf"))
        # Valid positions should be finite
        assert all(np.isfinite(result_np[mask_np]))

    def test_bypass_returns_masked(self) -> None:
        from tgirl.transport_mlx import redistribute_logits_mlx

        logits = mx.array([1.0, 2.0, 3.0])
        mask = mx.array([True, True, True])
        embeddings = mx.array(np.random.randn(3, 8).astype(np.float32))
        result = redistribute_logits_mlx(logits, mask, embeddings)
        assert result.bypassed is True
        assert result.bypass_reason == "valid_ratio_high"


class TestZeroCoupling:
    """transport_mlx.py only imports TransportConfig from tgirl, nothing else."""

    def test_only_transport_config_imported(self) -> None:
        import inspect

        import tgirl.transport_mlx as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        tgirl_imports = [
            line.strip()
            for line in lines
            if ("from tgirl" in line or "import tgirl" in line)
            and not line.strip().startswith("#")
        ]
        # Should only have one import: TransportConfig from transport
        assert len(tgirl_imports) == 1
        assert "TransportConfig" in tgirl_imports[0]

    def test_no_torch_import(self) -> None:
        """Module should not import torch (zero coupling)."""
        import inspect

        import tgirl.transport_mlx as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        torch_imports = [
            line.strip()
            for line in lines
            if ("import torch" in line)
            and not line.strip().startswith("#")
        ]
        assert torch_imports == [], (
            f"transport_mlx.py imports torch: {torch_imports}"
        )
