# PRP: tgirl.transport — Optimal Transport Logit Redistribution

## Source PRD: docs/PRDs/transport.md
## Date: 2026-03-12

## 1. Context Summary

Standard constrained generation masks invalid tokens and renormalizes, losing information about the model's intent. `tgirl.transport` redistributes probability mass from invalid to valid tokens using optimal transport, with cost defined by semantic distance in embedding space. The model's intent is preserved as much as the grammar allows.

This is the fourth core module (after registry, grammar, compile). It has **zero coupling** to any other tgirl module — operates on raw tensors only. Per CLAUDE.md, `transport.py` is security-sensitive (numerical correctness).

**Ordering constraint:** The sampler zeros out tokens for exhausted quotas/budgets in the valid_mask *before* calling transport. Transport receives a pre-masked valid_mask and never redistributes mass toward quota-exhausted tokens.

## 2. Codebase Analysis

### Relevant existing patterns
- Frozen Pydantic models for config types (see `CompileConfig`, `GrammarConfig`)
- structlog for logging
- Private helpers prefixed with `_`
- Test classes grouped by feature (`class TestXxx:`)
- `pytest.fixture()` for shared setup
- `pyproject.toml:24` — `transport = ["torch>=2.0", "pot>=0.9"]` already declared

### Key design decisions
- **Custom log-domain Sinkhorn over POT.** ~30 lines of torch code. Full control over convergence, Wasserstein extraction, GPU compat. POT stays as test validation dependency.
- **Log-domain Sinkhorn for numerical stability.** Uses `log_K = -C/epsilon` and `logsumexp` instead of `K = exp(-C/epsilon)`.
- **TransportResult as NamedTuple.** Unpacks as `(logits, distance)` for spec. Named access for telemetry fields.
- **Max 20 Sinkhorn iterations.** Typical convergence in 5-10.
- **Return log-space logits.** Downstream expects logits. Invalid tokens get `-inf`.

## 3. Module Structure

```python
# src/tgirl/transport.py — zero tgirl imports

class TransportConfig(BaseModel):       # frozen
    epsilon: float = 0.1
    max_iterations: int = 20
    convergence_threshold: float = 1e-6
    valid_ratio_threshold: float = 0.5
    invalid_mass_threshold: float = 0.01

class TransportResult(NamedTuple):
    logits: torch.Tensor
    wasserstein_distance: float
    bypassed: bool
    bypass_reason: str | None
    iterations: int

def redistribute_logits(logits, valid_mask, embeddings,
                        epsilon=0.1, max_iterations=20,
                        convergence_threshold=1e-6,
                        config=None) -> TransportResult

def _check_bypass(logits, valid_mask, config) -> tuple[bool, str | None]
def _standard_masking(logits, valid_mask) -> torch.Tensor
def _compute_cost_submatrix(embeddings, invalid_idx, valid_idx) -> torch.Tensor
def _sinkhorn_log_domain(cost, source_mass, target_capacity,
                          epsilon, max_iter, threshold) -> tuple[plan, w_dist, iters]
def _apply_transport_plan(plan, valid_idx, original_logits, vocab_size) -> torch.Tensor
```

## 4. Files

| File | Action |
|------|--------|
| `src/tgirl/transport.py` | Create — all transport logic |
| `tests/test_transport.py` | Create — unit tests |
| `tests/test_integration_transport.py` | Create — integration tests |
| `src/tgirl/__init__.py` | Update — add transport exports |

## 5. Implementation Tasks

### Task 1: Module skeleton, TransportConfig, TransportResult

Define types. TransportConfig is frozen Pydantic. TransportResult is NamedTuple that unpacks as 2-tuple for spec compatibility.

**Test Command:** `pytest tests/test_transport.py::TestTransportConfig tests/test_transport.py::TestTransportResult -v`

**Tests:** Config frozen, defaults match spec, custom values, Result tuple unpacking, Result named access, **zero tgirl imports in transport.py** (grep verification).

### Task 2: Bypass condition detection

`_check_bypass(logits, valid_mask, config)` → `(should_bypass, reason)`.

Three conditions checked in order:
1. `valid_mask.sum() <= 1` → `"forced_decode"`
2. `valid_mask.float().mean() > config.valid_ratio_threshold` → `"valid_ratio_high"`
3. Invalid probability mass < `config.invalid_mass_threshold` → `"invalid_mass_low"` (requires softmax computation)

**Test Command:** `pytest tests/test_transport.py::TestCheckBypass -v`

**Tests:** Each condition triggers/doesn't trigger, priority ordering, custom thresholds, edge cases (0 valid, all valid).

### Task 3: Standard masking fallback

`_standard_masking(logits, valid_mask)` — sets invalid logits to `-inf`, preserves valid logits unchanged. This is the bypass fast path.

**Test Command:** `pytest tests/test_transport.py::TestStandardMasking -v`

**Tests:** Invalid → `-inf`, valid unchanged, output shape, all-valid identity, single valid.

### Task 4: Cost submatrix computation

`_compute_cost_submatrix(embeddings, invalid_indices, valid_indices)` → `(n_invalid, n_valid)` tensor.

Cost = `1 - cosine_similarity`. Uses `F.normalize` + `matmul`. Only allocates the submatrix, never V×V.

**Test Command:** `pytest tests/test_transport.py::TestCostSubmatrix -v`

**Tests:** Identical embeddings → cost 0, orthogonal → cost 1, opposite → cost 2, output shape, values in [0, 2], various embed_dim.

### Task 5: Log-domain Sinkhorn

`_sinkhorn_log_domain(cost_matrix, source_mass, target_capacity, epsilon, max_iterations, convergence_threshold)` → `(transport_plan, wasserstein_distance, iterations)`.

Core algorithm:
- `log_K = -cost / epsilon`
- Iterate `log_u`, `log_v` scaling vectors with `logsumexp`
- Convergence on marginal error
- Wasserstein = `(plan * cost).sum()` from converged plan

**Test Command:** `pytest tests/test_transport.py::TestSinkhornLogDomain -v`

**Tests:** Uniform cost → uniform plan, zero cost → concentrated, convergence < max_iter, correct iteration count, non-negative Wasserstein, marginals match inputs (within tolerance), small vs large epsilon behavior, single source/target edges, **Hypothesis property test for marginal conservation**.

### Task 6: Transport plan application

`_apply_transport_plan(plan, valid_indices, original_logits, vocab_size)` → redistributed log-space logits.

Steps: sum plan columns → redistributed prob per valid token, add existing valid probs, convert to log-space, invalid → `-inf`.

**Test Command:** `pytest tests/test_transport.py::TestApplyTransportPlan -v`

**Tests:** Output shape `(vocab_size,)`, invalid → `-inf`, valid → finite, softmax sums to ~1.0, mass conservation, output is log-space.

### Task 7: Main `redistribute_logits` integration

Wire bypass → cost submatrix → Sinkhorn → plan application. Accept both individual params and TransportConfig.

**Test Command:** `pytest tests/test_transport.py::TestRedistributeLogits -v`

**Tests:** Bypass path returns masked logits + `bypassed=True`, full OT path returns redistributed + `bypassed=False`, invalid always `-inf`, probs sum to ~1, non-negative Wasserstein, high epsilon ≈ standard masking, low epsilon → concentrated on nearest valid, structlog events emitted, input tensors not mutated, CPU works, config object accepted.

### Task 8: Exports and integration tests

Update `__init__.py` exports. Write integration tests with larger vocabs (1000-5000), end-to-end random inputs, zero-coupling verification.

**Test Command:** `pytest tests/test_integration_transport.py -v`

**Tests:** Import paths work, module docstring exists, zero coupling verified, larger vocab stress tests, telemetry-compatible outputs.

## 6. Verification

```bash
pytest tests/test_transport.py tests/test_integration_transport.py -v
ruff check src/tgirl/transport.py tests/test_transport.py tests/test_integration_transport.py
```
