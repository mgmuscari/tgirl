# Plan Review: initial-scaffolding-registry

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-12
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. `typing.Any` not representable in `TypeRepr`
**Severity:** Structural (HIGH)
**Evidence:** The spec's canonical decorator example at TGIRL.md line 194 uses `dict[str, Any]`. The PRP's `TypeRepr` union had no `AnyType` variant and `extract_type` would raise `TypeError` on `Any`. No test covered this path.
**Proposer Response:** Accepted — added `AnyType` frozen dataclass to `TypeRepr`, added `typing.Any` handling in extraction logic, added three test cases (`test_any_type_is_frozen`, `test_extract_any`, `test_extract_dict_str_any`). Grammar-level semantics correctly deferred.
**PRP Updated:** Yes (Tasks 2, 3)

### 2. Mutable `dict` in frozen `RegistrySnapshot` breaks immutability
**Severity:** Moderate (MEDIUM)
**Evidence:** `RegistrySnapshot.quotas: dict[str, int]` inside `frozen=True` dataclass — dict reference is frozen but contents are mutable. `snapshot.quotas["tool"] = 999` succeeds silently. Contradicts "safety by construction" design principle.
**Proposer Response:** Accepted — changed `quotas` to use `types.MappingProxyType` wrapped via `__post_init__`, typed as `Mapping[str, int]`. Added `test_registry_snapshot_quotas_immutable`. Resolved Uncertainty Log item #2.
**PRP Updated:** Yes (Task 2, Uncertainty Log)

### 3. Single-scope per tool limits real authorization
**Severity:** Moderate (MEDIUM)
**Evidence:** `scope: str | None` (singular) cannot express tools requiring multiple scopes. Spec is explicit about single-scope (TGIRL.md line 189), but limitation was undocumented.
**Proposer Response:** Partially accepted — added Uncertainty Log item #5 documenting limitation with migration path (`scopes: frozenset[str] | None`), flagged for spec review before grammar module PRP. API change correctly rejected (PRP implements spec faithfully).
**PRP Updated:** Yes (Uncertainty Log)

### 4. Private `_type_extract.py` hides potentially useful API
**Severity:** Minor (LOW)
**Evidence:** Task 3 creates extraction module as private. Downstream modules like grammar might need direct access.
**Proposer Response:** Rejected — private is correct default; grammar consumes `TypeRepr` from snapshots, never calls `extract_type` directly; `TypeRepr` types themselves are public; separate file is for test organization, not API surface.
**PRP Updated:** No

### 5. No thread safety consideration for `ToolRegistry`
**Severity:** Minor (LOW)
**Evidence:** `ToolRegistry._tools` is a mutable dict with no synchronization. In a server context, concurrent access is possible.
**Proposer Response:** Accepted (documentation only) — added Uncertainty Log item #6 documenting that `ToolRegistry` is not thread-safe, registration is startup-time, `serve.py` must register all tools before starting. GIL provides adequate protection for intended usage.
**PRP Updated:** Yes (Uncertainty Log)

## What Holds Well

1. **Thorough test matrix.** 57+ individual test cases across 5 tasks with clear RED-first TDD structure. Type extraction coverage is particularly comprehensive (22 cases covering every type path).
2. **Correct module isolation.** Dependency graph respected — `tgirl.types` and `tgirl.registry` are stdlib-only, with an explicit test for this.
3. **Well-structured task decomposition.** Five atomic tasks with clear dependency ordering (scaffolding → types → extraction → registry → integration).
4. **Honest uncertainty log.** Identified real issues proactively — two yield points built on items already flagged there.
5. **Determinism as first-class concern.** Snapshot determinism (sorted tools, explicit timestamp handling) is specified and tested, critical for downstream grammar diffing.

## Summary

The PRP is structurally sound. All HIGH and MEDIUM concerns were addressed with concrete changes (AnyType variant, MappingProxyType for immutability, documented scope limitation). Two LOW-severity items were handled appropriately — one rejected with well-defended justification, one accepted as documentation. The revised PRP is ready for implementation.
