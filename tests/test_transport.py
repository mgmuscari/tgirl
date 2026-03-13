"""Tests for tgirl.transport — Optimal transport logit redistribution."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError


class TestTransportConfig:
    """Task 1: TransportConfig is a frozen Pydantic model with correct defaults."""

    def test_config_is_frozen(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig()
        with pytest.raises(ValidationError):
            c.epsilon = 0.5  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig()
        assert c.epsilon == 0.1
        assert c.max_iterations == 20
        assert c.convergence_threshold == 1e-6
        assert c.valid_ratio_threshold == 0.5
        assert c.invalid_mass_threshold == 0.01

    def test_config_custom_values(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig(
            epsilon=0.05,
            max_iterations=50,
            convergence_threshold=1e-8,
            valid_ratio_threshold=0.3,
            invalid_mass_threshold=0.05,
        )
        assert c.epsilon == 0.05
        assert c.max_iterations == 50
        assert c.convergence_threshold == 1e-8
        assert c.valid_ratio_threshold == 0.3
        assert c.invalid_mass_threshold == 0.05


class TestTransportResult:
    """Task 1: TransportResult is a NamedTuple with 2-tuple unpacking."""

    def test_result_tuple_unpacking_positional(self) -> None:
        """Spec compatibility: first two positional fields are logits, distance."""
        from tgirl.transport import TransportResult

        logits = torch.tensor([1.0, 2.0])
        r = TransportResult(
            logits=logits,
            wasserstein_distance=0.5,
            bypassed=False,
            bypass_reason=None,
            iterations=10,
        )
        assert torch.equal(r[0], logits)
        assert r[1] == 0.5

    def test_result_named_access(self) -> None:
        from tgirl.transport import TransportResult

        logits = torch.tensor([1.0, 2.0])
        r = TransportResult(
            logits=logits,
            wasserstein_distance=0.5,
            bypassed=True,
            bypass_reason="forced_decode",
            iterations=0,
        )
        assert torch.equal(r.logits, logits)
        assert r.wasserstein_distance == 0.5
        assert r.bypassed is True
        assert r.bypass_reason == "forced_decode"
        assert r.iterations == 0

    def test_result_is_namedtuple(self) -> None:
        from tgirl.transport import TransportResult

        assert hasattr(TransportResult, "_fields")
        assert "logits" in TransportResult._fields
        assert "wasserstein_distance" in TransportResult._fields


class TestZeroCoupling:
    """Task 1: transport.py must not import from tgirl."""

    def test_no_tgirl_imports(self) -> None:
        """Verify transport.py has zero tgirl imports via source inspection."""
        import inspect

        import tgirl.transport as mod

        source = inspect.getsource(mod)
        # Should not contain any 'from tgirl' or 'import tgirl'
        lines = source.split("\n")
        tgirl_imports = [
            line
            for line in lines
            if ("from tgirl" in line or "import tgirl" in line)
            and not line.strip().startswith("#")
        ]
        assert tgirl_imports == [], (
            f"transport.py has tgirl imports: {tgirl_imports}"
        )


class TestCheckBypass:
    """Task 2: _check_bypass detects when OT should be skipped."""

    def test_forced_decode_zero_valid(self) -> None:
        """No valid tokens → forced_decode."""
        from tgirl.transport import TransportConfig, _check_bypass

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, False, False])
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is True
        assert reason == "forced_decode"

    def test_forced_decode_one_valid(self) -> None:
        """Single valid token → forced_decode."""
        from tgirl.transport import TransportConfig, _check_bypass

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, True, False])
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is True
        assert reason == "forced_decode"

    def test_valid_ratio_high(self) -> None:
        """Most tokens valid → valid_ratio_high."""
        from tgirl.transport import TransportConfig, _check_bypass

        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # 3 out of 4 valid = 0.75 > default 0.5
        mask = torch.tensor([True, True, True, False])
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is True
        assert reason == "valid_ratio_high"

    def test_invalid_mass_low(self) -> None:
        """Very little probability on invalid tokens → invalid_mass_low."""
        from tgirl.transport import TransportConfig, _check_bypass

        # 2 valid out of 5 = 0.4 (below ratio threshold 0.5)
        # But most mass on valid tokens, negligible on invalid
        logits = torch.tensor([10.0, 10.0, -100.0, -100.0, -100.0])
        mask = torch.tensor([True, True, False, False, False])
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is True
        assert reason == "invalid_mass_low"

    def test_no_bypass(self) -> None:
        """Significant invalid mass, low valid ratio → no bypass."""
        from tgirl.transport import TransportConfig, _check_bypass

        # 2 valid out of 5 = 0.4 < 0.5, and significant invalid mass
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        mask = torch.tensor([True, True, False, False, False])
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is False
        assert reason is None

    def test_priority_forced_decode_before_ratio(self) -> None:
        """forced_decode takes priority even when ratio is high."""
        from tgirl.transport import TransportConfig, _check_bypass

        logits = torch.tensor([1.0])
        mask = torch.tensor([True])  # 1 valid, but <= 1
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is True
        assert reason == "forced_decode"

    def test_custom_thresholds(self) -> None:
        """Custom thresholds change bypass behavior."""
        from tgirl.transport import TransportConfig, _check_bypass

        # 2 out of 4 valid = 0.5, with threshold at 0.3 → triggers
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        mask = torch.tensor([True, True, False, False])
        config = TransportConfig(valid_ratio_threshold=0.3)
        should, reason = _check_bypass(logits, mask, config)
        assert should is True
        assert reason == "valid_ratio_high"

    def test_all_valid_triggers_ratio(self) -> None:
        """All tokens valid → valid_ratio_high (after forced_decode check)."""
        from tgirl.transport import TransportConfig, _check_bypass

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, True, True])
        should, reason = _check_bypass(logits, mask, TransportConfig())
        assert should is True
        assert reason == "valid_ratio_high"


class TestStandardMasking:
    """Task 3: _standard_masking sets invalid logits to -inf."""

    def test_invalid_become_neg_inf(self) -> None:
        from tgirl.transport import _standard_masking

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        result = _standard_masking(logits, mask)
        assert result[1] == float("-inf")

    def test_valid_unchanged(self) -> None:
        from tgirl.transport import _standard_masking

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        result = _standard_masking(logits, mask)
        assert result[0] == 1.0
        assert result[2] == 3.0

    def test_output_shape(self) -> None:
        from tgirl.transport import _standard_masking

        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        result = _standard_masking(logits, mask)
        assert result.shape == logits.shape

    def test_all_valid_identity(self) -> None:
        from tgirl.transport import _standard_masking

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, True, True])
        result = _standard_masking(logits, mask)
        assert torch.equal(result, logits)

    def test_single_valid(self) -> None:
        from tgirl.transport import _standard_masking

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, True, False])
        result = _standard_masking(logits, mask)
        assert result[1] == 2.0
        assert result[0] == float("-inf")
        assert result[2] == float("-inf")

    def test_does_not_mutate_input(self) -> None:
        from tgirl.transport import _standard_masking

        logits = torch.tensor([1.0, 2.0, 3.0])
        original = logits.clone()
        mask = torch.tensor([True, False, True])
        _standard_masking(logits, mask)
        assert torch.equal(logits, original)


class TestCostSubmatrix:
    """Task 4: _compute_cost_submatrix computes cosine distance submatrix."""

    def test_identical_embeddings_cost_zero(self) -> None:
        from tgirl.transport import _compute_cost_submatrix

        # All embeddings identical → cost = 0
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        invalid_idx = torch.tensor([0])
        valid_idx = torch.tensor([1, 2])
        cost = _compute_cost_submatrix(embeddings, invalid_idx, valid_idx)
        assert torch.allclose(cost, torch.zeros(1, 2), atol=1e-6)

    def test_orthogonal_embeddings_cost_one(self) -> None:
        from tgirl.transport import _compute_cost_submatrix

        # Orthogonal → cost = 1
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        invalid_idx = torch.tensor([0])
        valid_idx = torch.tensor([1])
        cost = _compute_cost_submatrix(embeddings, invalid_idx, valid_idx)
        assert torch.allclose(cost, torch.ones(1, 1), atol=1e-6)

    def test_opposite_embeddings_cost_two(self) -> None:
        from tgirl.transport import _compute_cost_submatrix

        # Opposite → cost = 2
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        invalid_idx = torch.tensor([0])
        valid_idx = torch.tensor([1])
        cost = _compute_cost_submatrix(embeddings, invalid_idx, valid_idx)
        assert torch.allclose(cost, torch.tensor([[2.0]]), atol=1e-6)

    def test_output_shape(self) -> None:
        from tgirl.transport import _compute_cost_submatrix

        embeddings = torch.randn(10, 64)
        invalid_idx = torch.tensor([0, 2, 5])
        valid_idx = torch.tensor([1, 3, 4, 6, 7, 8, 9])
        cost = _compute_cost_submatrix(embeddings, invalid_idx, valid_idx)
        assert cost.shape == (3, 7)

    def test_values_in_range(self) -> None:
        from tgirl.transport import _compute_cost_submatrix

        torch.manual_seed(42)
        embeddings = torch.randn(20, 32)
        invalid_idx = torch.tensor([0, 1, 2, 3, 4])
        valid_idx = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        cost = _compute_cost_submatrix(embeddings, invalid_idx, valid_idx)
        assert (cost >= -1e-6).all(), "Cost should be non-negative"
        assert (cost <= 2.0 + 1e-6).all(), "Cost should be at most 2"

    def test_various_embed_dim(self) -> None:
        from tgirl.transport import _compute_cost_submatrix

        for dim in [1, 8, 128, 512]:
            embeddings = torch.randn(5, dim)
            invalid_idx = torch.tensor([0, 1])
            valid_idx = torch.tensor([2, 3, 4])
            cost = _compute_cost_submatrix(embeddings, invalid_idx, valid_idx)
            assert cost.shape == (2, 3)


class TestSinkhornLogDomain:
    """Task 5: _sinkhorn_log_domain optimal transport solver."""

    def test_uniform_cost_uniform_plan(self) -> None:
        """Uniform cost matrix → plan should be approximately uniform."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.ones(3, 3)
        source = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
        target = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
        plan, w_dist, iters = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.1, max_iterations=100,
            convergence_threshold=1e-8,
        )
        # Plan should be uniform: each cell ≈ 1/9
        expected = torch.full((3, 3), 1.0 / 9)
        assert torch.allclose(plan, expected, atol=1e-4)

    def test_zero_cost_concentrated(self) -> None:
        """Zero cost on diagonal → mass concentrates on diagonal."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.ones(2, 2)
        cost[0, 0] = 0.0
        cost[1, 1] = 0.0
        source = torch.tensor([0.5, 0.5])
        target = torch.tensor([0.5, 0.5])
        plan, w_dist, iters = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.01, max_iterations=100,
            convergence_threshold=1e-8,
        )
        # Most mass should be on diagonal
        assert plan[0, 0] > plan[0, 1]
        assert plan[1, 1] > plan[1, 0]

    def test_convergence_before_max_iter(self) -> None:
        """Should converge in fewer than max iterations."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.rand(3, 4)
        source = torch.tensor([0.3, 0.3, 0.4])
        target = torch.tensor([0.2, 0.3, 0.2, 0.3])
        _, _, iters = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.1, max_iterations=100,
            convergence_threshold=1e-6,
        )
        assert iters < 100

    def test_correct_iteration_count(self) -> None:
        """Returned iteration count should be positive."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.rand(2, 3)
        source = torch.tensor([0.5, 0.5])
        target = torch.tensor([0.3, 0.3, 0.4])
        _, _, iters = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.1, max_iterations=20,
            convergence_threshold=1e-6,
        )
        assert iters >= 1

    def test_non_negative_wasserstein(self) -> None:
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.rand(3, 4)
        source = torch.tensor([0.3, 0.3, 0.4])
        target = torch.tensor([0.2, 0.3, 0.2, 0.3])
        _, w_dist, _ = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.1, max_iterations=50,
            convergence_threshold=1e-6,
        )
        assert w_dist >= 0.0

    def test_marginals_match(self) -> None:
        """Row sums ≈ source, column sums ≈ target."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.rand(3, 4)
        source = torch.tensor([0.3, 0.3, 0.4])
        target = torch.tensor([0.2, 0.3, 0.2, 0.3])
        plan, _, _ = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.1, max_iterations=100,
            convergence_threshold=1e-8,
        )
        assert torch.allclose(plan.sum(dim=1), source, atol=1e-4)
        assert torch.allclose(plan.sum(dim=0), target, atol=1e-4)

    def test_small_epsilon_more_concentrated(self) -> None:
        """Smaller epsilon → plan more concentrated on low-cost entries."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.tensor([[0.1, 1.0], [1.0, 0.1]])
        source = torch.tensor([0.5, 0.5])
        target = torch.tensor([0.5, 0.5])
        plan_small, _, _ = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.01, max_iterations=100,
            convergence_threshold=1e-8,
        )
        plan_large, _, _ = _sinkhorn_log_domain(
            cost, source, target, epsilon=1.0, max_iterations=100,
            convergence_threshold=1e-8,
        )
        # Small epsilon: more mass on diagonal (low cost)
        assert plan_small[0, 0] > plan_large[0, 0]

    def test_single_source_target(self) -> None:
        """1x1 transport plan."""
        from tgirl.transport import _sinkhorn_log_domain

        cost = torch.tensor([[0.5]])
        source = torch.tensor([1.0])
        target = torch.tensor([1.0])
        plan, w_dist, iters = _sinkhorn_log_domain(
            cost, source, target, epsilon=0.1, max_iterations=20,
            convergence_threshold=1e-6,
        )
        assert torch.allclose(plan, torch.tensor([[1.0]]), atol=1e-4)
        assert abs(w_dist - 0.5) < 1e-4

    def test_hypothesis_marginal_conservation(self) -> None:
        """Property test: marginals are always conserved."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from tgirl.transport import _sinkhorn_log_domain

        @given(
            n_source=st.integers(min_value=1, max_value=5),
            n_target=st.integers(min_value=1, max_value=5),
            seed=st.integers(min_value=0, max_value=10000),
        )
        @settings(max_examples=20, deadline=5000)
        def _check(n_source: int, n_target: int, seed: int) -> None:
            torch.manual_seed(seed)
            cost = torch.rand(n_source, n_target)
            # Create valid probability distributions
            source_raw = torch.rand(n_source) + 0.1
            source = source_raw / source_raw.sum()
            target_raw = torch.rand(n_target) + 0.1
            target = target_raw / target_raw.sum()

            plan, _, _ = _sinkhorn_log_domain(
                cost, source, target, epsilon=0.1,
                max_iterations=100, convergence_threshold=1e-6,
            )
            assert torch.allclose(plan.sum(dim=1), source, atol=1e-3)
            assert torch.allclose(plan.sum(dim=0), target, atol=1e-3)

        _check()
