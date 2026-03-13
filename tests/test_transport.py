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

    def test_epsilon_zero_rejected(self) -> None:
        from tgirl.transport import TransportConfig

        with pytest.raises(ValidationError):
            TransportConfig(epsilon=0)

    def test_epsilon_negative_rejected(self) -> None:
        from tgirl.transport import TransportConfig

        with pytest.raises(ValidationError):
            TransportConfig(epsilon=-1)

    def test_max_iterations_too_high_rejected(self) -> None:
        from tgirl.transport import TransportConfig

        with pytest.raises(ValidationError):
            TransportConfig(max_iterations=1001)

    def test_valid_ratio_out_of_range_rejected(self) -> None:
        from tgirl.transport import TransportConfig

        with pytest.raises(ValidationError):
            TransportConfig(valid_ratio_threshold=-0.5)
        with pytest.raises(ValidationError):
            TransportConfig(valid_ratio_threshold=1.5)

    def test_valid_defaults_still_accepted(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig()
        assert c.epsilon == 0.1

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


class TestApplyTransportPlan:
    """Task 6: _apply_transport_plan converts plan to redistributed logits."""

    def _make_plan_and_indices(self) -> tuple:
        """Helper: create a simple transport plan scenario."""
        # Vocab size 5, tokens 0,2 invalid, tokens 1,3,4 valid
        # Invalid token 0 has prob 0.3, invalid token 2 has prob 0.2
        # Valid tokens 1,3,4 have probs 0.2, 0.15, 0.15
        valid_idx = torch.tensor([1, 3, 4])
        # Plan: 2 invalid sources → 3 valid targets
        plan = torch.tensor([
            [0.15, 0.10, 0.05],  # invalid 0 sends mass
            [0.05, 0.05, 0.10],  # invalid 2 sends mass
        ])
        original_logits = torch.log(
            torch.tensor([0.30, 0.20, 0.20, 0.15, 0.15])
        )
        return plan, valid_idx, original_logits, 5

    def test_output_shape(self) -> None:
        from tgirl.transport import _apply_transport_plan

        plan, valid_idx, logits, vocab_size = self._make_plan_and_indices()
        result = _apply_transport_plan(plan, valid_idx, logits, vocab_size)
        assert result.shape == (vocab_size,)

    def test_invalid_neg_inf(self) -> None:
        from tgirl.transport import _apply_transport_plan

        plan, valid_idx, logits, vocab_size = self._make_plan_and_indices()
        result = _apply_transport_plan(plan, valid_idx, logits, vocab_size)
        # Tokens 0 and 2 are invalid → -inf
        assert result[0] == float("-inf")
        assert result[2] == float("-inf")

    def test_valid_finite(self) -> None:
        from tgirl.transport import _apply_transport_plan

        plan, valid_idx, logits, vocab_size = self._make_plan_and_indices()
        result = _apply_transport_plan(plan, valid_idx, logits, vocab_size)
        # Valid tokens 1, 3, 4 should be finite
        assert torch.isfinite(result[1])
        assert torch.isfinite(result[3])
        assert torch.isfinite(result[4])

    def test_softmax_sums_to_one(self) -> None:
        from tgirl.transport import _apply_transport_plan

        plan, valid_idx, logits, vocab_size = self._make_plan_and_indices()
        result = _apply_transport_plan(plan, valid_idx, logits, vocab_size)
        # Softmax of finite values should sum to ~1
        finite_mask = torch.isfinite(result)
        probs = torch.softmax(result[finite_mask], dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_mass_conservation(self) -> None:
        from tgirl.transport import _apply_transport_plan

        plan, valid_idx, logits, vocab_size = self._make_plan_and_indices()
        result = _apply_transport_plan(plan, valid_idx, logits, vocab_size)
        # Total redistributed prob should equal original valid + transported
        original_probs = torch.softmax(logits, dim=-1)
        original_valid_mass = original_probs[valid_idx].sum().item()
        transported_mass = plan.sum().item()
        total_expected = original_valid_mass + transported_mass
        # Result probs (from finite logits)
        finite_mask = torch.isfinite(result)
        result_probs = torch.exp(result[finite_mask])
        assert abs(result_probs.sum().item() - total_expected) < 1e-4

    def test_output_is_log_space(self) -> None:
        from tgirl.transport import _apply_transport_plan

        plan, valid_idx, logits, vocab_size = self._make_plan_and_indices()
        result = _apply_transport_plan(plan, valid_idx, logits, vocab_size)
        # Valid logits should be negative (log of probability < 1)
        valid_logits = result[valid_idx]
        assert (valid_logits < 0).all()


class TestRedistributeLogits:
    """Task 7: Main redistribute_logits wires bypass -> OT -> application."""

    @pytest.fixture()
    def simple_setup(self) -> dict:
        """Create a simple test scenario: 10 tokens, 4 valid."""
        torch.manual_seed(42)
        vocab_size = 10
        embed_dim = 16
        logits = torch.randn(vocab_size)
        valid_mask = torch.tensor(
            [True, True, False, False, True, False, False, True, False, False]
        )
        embeddings = torch.randn(vocab_size, embed_dim)
        return {
            "logits": logits,
            "valid_mask": valid_mask,
            "embeddings": embeddings,
        }

    def test_bypass_returns_masked_logits(self) -> None:
        """When bypass triggers, result has bypassed=True."""
        from tgirl.transport import redistribute_logits

        logits = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, True, True])  # all valid → ratio high
        embeddings = torch.randn(3, 8)
        result = redistribute_logits(logits, mask, embeddings)
        assert result.bypassed is True
        assert result.bypass_reason == "valid_ratio_high"
        assert result.iterations == 0

    def test_full_ot_path(self, simple_setup: dict) -> None:
        """Full OT path returns bypassed=False with valid results."""
        from tgirl.transport import redistribute_logits

        result = redistribute_logits(**simple_setup)
        assert result.bypassed is False
        assert result.bypass_reason is None
        assert result.iterations > 0

    def test_invalid_always_neg_inf(self, simple_setup: dict) -> None:
        from tgirl.transport import redistribute_logits

        result = redistribute_logits(**simple_setup)
        mask = simple_setup["valid_mask"]
        invalid_logits = result.logits[~mask]
        assert (invalid_logits == float("-inf")).all()

    def test_probs_sum_to_one(self, simple_setup: dict) -> None:
        from tgirl.transport import redistribute_logits

        result = redistribute_logits(**simple_setup)
        finite = result.logits[torch.isfinite(result.logits)]
        probs = torch.softmax(finite, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_non_negative_wasserstein(self, simple_setup: dict) -> None:
        from tgirl.transport import redistribute_logits

        result = redistribute_logits(**simple_setup)
        assert result.wasserstein_distance >= 0.0

    def test_high_epsilon_approx_standard_masking(self) -> None:
        """Very high epsilon ≈ uniform redistribution ≈ standard masking behavior."""
        from tgirl.transport import TransportConfig, redistribute_logits

        torch.manual_seed(123)
        logits = torch.randn(10)
        mask = torch.tensor(
            [True, True, False, False, True, False, False, True, False, False]
        )
        embeddings = torch.randn(10, 16)
        config = TransportConfig(epsilon=100.0)
        result = redistribute_logits(logits, mask, embeddings, config=config)
        # With very high epsilon, all valid tokens get roughly equal share
        # of redistributed mass — but this is still different from standard masking
        assert result.bypassed is False
        assert (result.logits[~mask] == float("-inf")).all()

    def test_input_tensors_not_mutated(self, simple_setup: dict) -> None:
        from tgirl.transport import redistribute_logits

        original_logits = simple_setup["logits"].clone()
        original_mask = simple_setup["valid_mask"].clone()
        original_emb = simple_setup["embeddings"].clone()
        redistribute_logits(**simple_setup)
        assert torch.equal(simple_setup["logits"], original_logits)
        assert torch.equal(simple_setup["valid_mask"], original_mask)
        assert torch.equal(simple_setup["embeddings"], original_emb)

    def test_config_object_accepted(self) -> None:
        from tgirl.transport import TransportConfig, redistribute_logits

        torch.manual_seed(42)
        logits = torch.randn(10)
        mask = torch.tensor(
            [True, True, False, False, True, False, False, True, False, False]
        )
        embeddings = torch.randn(10, 16)
        config = TransportConfig(epsilon=0.05, max_iterations=10)
        result = redistribute_logits(logits, mask, embeddings, config=config)
        assert isinstance(result.logits, torch.Tensor)

    def test_extreme_logits_no_neg_inf_for_valid(self) -> None:
        """F3: extreme logits must not produce -inf for valid tokens."""
        from tgirl.transport import redistribute_logits

        torch.manual_seed(42)
        # One valid token dominates (logit 100), another valid underflows
        # (logit -1000). Invalid tokens have matching high logits to keep
        # invalid_mass high enough to avoid bypass.
        logits = torch.tensor([100.0, -1000.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0])
        valid_mask = torch.tensor([True, True, False, False, False,
                                   False, False, False, False, False])
        embeddings = torch.randn(10, 16)
        result = redistribute_logits(logits, valid_mask, embeddings)
        assert not result.bypassed, "Should exercise full OT path"
        # Both valid tokens must have finite output (not -inf)
        assert torch.isfinite(result.logits[0]), "Valid token 0 should be finite"
        assert torch.isfinite(result.logits[1]), "Valid token 1 should be finite"

    def test_extreme_logit_gap_mass_conservation(self) -> None:
        """F3: softmax of output sums to ~1.0 even with extreme logit gaps."""
        from tgirl.transport import redistribute_logits

        torch.manual_seed(42)
        logits = torch.tensor([100.0, -1000.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 100.0, 100.0, 100.0])
        valid_mask = torch.tensor([True, True, False, False, False,
                                   False, False, False, False, False])
        embeddings = torch.randn(10, 16)
        result = redistribute_logits(logits, valid_mask, embeddings)
        finite = result.logits[torch.isfinite(result.logits)]
        probs = torch.softmax(finite, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_structlog_events_emitted(self, simple_setup: dict) -> None:
        """Structlog events should be emitted during redistribution."""
        from structlog.testing import capture_logs

        from tgirl.transport import redistribute_logits

        with capture_logs() as logs:
            redistribute_logits(**simple_setup)
        # Should have at least one log event
        assert len(logs) > 0
