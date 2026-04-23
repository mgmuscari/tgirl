"""Offline calibration pipeline for ESTRADIOL v2.

Discovers the bottleneck layer, extracts behavioral vectors via
contrastive generation, builds the low-rank codebook via SVD,
and validates scaffold/complement decomposition.

Adapted from platonic-experiments (experiments 27a, 27d, 27c).
"""

from __future__ import annotations

import math
import time
from typing import Any

import mlx.core as mx
import structlog

from tgirl.format import TokenizerProto

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Effective rank metrics
# ---------------------------------------------------------------------------


def effective_rank(X: mx.array) -> float:
    """Effective rank via Shannon entropy of singular values.

    exp(-Sigma p_i log p_i) where p_i = s_i / Sigma_s.
    Returns 1.0 for rank-1, d for full rank (identity).

    Args:
        X: (n_samples, d_features) matrix.
    """
    _, S, _ = mx.linalg.svd(X, stream=mx.cpu)
    mx.eval(S)
    # Filter near-zero singular values
    s = [float(v) for v in S.tolist() if v > 1e-10]
    if not s:
        return 0.0
    total = sum(s)
    entropy = -sum((si / total) * math.log(si / total) for si in s)
    return math.exp(entropy)


def participation_ratio(s: mx.array) -> float:
    """Participation ratio: (Sigma_s)^2 / Sigma(s^2).

    Used for codebook compression analysis (experiment27d).
    Returns 1.0 for single dominant value, N for uniform spectrum.
    """
    s_sum = float(mx.sum(s).item())
    s_sq_sum = float(mx.sum(s * s).item())
    if s_sq_sum < 1e-30:
        return 0.0
    return (s_sum * s_sum) / s_sq_sum


# ---------------------------------------------------------------------------
# Bottleneck discovery
# ---------------------------------------------------------------------------


class _AllPositionsHook:
    """Captures activations at ALL token positions for rank analysis.

    Unlike _BottleneckHook (which captures last-token only for runtime),
    this hook captures the full (seq_len, d_model) output at the target
    layer. Used only during calibration.
    """

    def __init__(self, layers: Any, layer_idx: int) -> None:
        self._target = layers[layer_idx]
        self._layer_type = type(self._target)
        self._original_call: Any = None
        self._captured: Any = None  # (seq_len, d_model)
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._original_call = self._layer_type.__call__
        hook = self

        def _patched(layer_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            result = hook._original_call(layer_self, x, *args, **kwargs)
            if layer_self is hook._target:
                # Capture all positions: (batch=1, seq_len, d_model) → (seq_len, d_model)
                hook._captured = result[0, :, :].astype(mx.float32)
            return result

        self._layer_type.__call__ = _patched
        self._installed = True

    def uninstall(self) -> None:
        if self._installed and self._original_call is not None:
            self._layer_type.__call__ = self._original_call
            self._installed = False
            self._original_call = None


def discover_bottleneck(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layer_path: str = "model.layers",
) -> tuple[int, list[float]]:
    """Sweep effective rank across all layers to find the bottleneck.

    For each layer, captures activations at ALL token positions across
    all texts, producing a (total_tokens, d_model) matrix. Effective
    rank of this matrix measures the dimensionality of the layer's
    representational space.

    Args:
        model: MLX model with make_cache() and __call__.
        tokenizer: Tokenizer with encode().
        texts: Calibration texts (10+ diverse sentences recommended).
        layer_path: Dot-separated path to model layers list.

    Returns:
        (bottleneck_layer_idx, effective_ranks_per_layer)
    """
    # Navigate to layers
    layers = model
    for attr in layer_path.split("."):
        layers = getattr(layers, attr)
    n_layers = len(layers)

    logger.info("discover_bottleneck_start", n_layers=n_layers, n_texts=len(texts))
    t0 = time.monotonic()

    ranks: list[float] = []
    for layer_idx in range(n_layers):
        hook = _AllPositionsHook(layers, layer_idx=layer_idx)
        hook.install()
        try:
            all_activations = []
            for text in texts:
                token_ids = tokenizer.encode(text)
                cache = model.make_cache()
                input_ids = mx.array([token_ids])
                _ = model(input_ids, cache=cache)
                if hook._captured is not None:
                    all_activations.append(hook._captured)
                hook._captured = None

            if all_activations:
                # Concatenate: (total_tokens, d_model)
                M = mx.concatenate(all_activations, axis=0)
                mx.eval(M)
                rank = effective_rank(M)
            else:
                rank = float("inf")
            ranks.append(rank)
        finally:
            hook.uninstall()

    bn_layer = int(min(range(n_layers), key=lambda i: ranks[i]))
    elapsed = time.monotonic() - t0
    logger.info(
        "discover_bottleneck_done",
        bottleneck_layer=bn_layer,
        bottleneck_rank=round(ranks[bn_layer], 2),
        elapsed_s=round(elapsed, 1),
    )
    return bn_layer, ranks


# ---------------------------------------------------------------------------
# Generation-based hook for contrastive extraction
# ---------------------------------------------------------------------------


class _GenerationHook:
    """Captures last-token activations during generation.

    Unlike _BottleneckHook (which captures a single activation per forward),
    this hook accumulates activations across multiple generation steps and
    returns their mean. Used for contrastive vector extraction.

    Ported from platonic-experiments experiment26n/LayerHook.
    """

    def __init__(self, layers: Any, layer_idx: int) -> None:
        self._target = layers[layer_idx]
        self._layer_type = type(self._target)
        self._original_call: Any = None
        self._captured: list[Any] = []
        self._active = False
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._original_call = self._layer_type.__call__
        hook = self

        def _patched(layer_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            result = hook._original_call(layer_self, x, *args, **kwargs)
            if hook._active and layer_self is hook._target:
                hook._captured.append(result[:, -1:, :].astype(mx.float32))
            return result

        self._layer_type.__call__ = _patched
        self._installed = True

    def uninstall(self) -> None:
        if self._installed and self._original_call is not None:
            self._layer_type.__call__ = self._original_call
            self._installed = False
            self._original_call = None

    def start_capture(self) -> None:
        self._captured = []
        self._active = True

    def stop(self) -> None:
        self._active = False

    def mean_state(self) -> mx.array:
        """Mean activation across all captured tokens → (d_model,)."""
        if not self._captured:
            msg = "No activations captured"
            raise RuntimeError(msg)
        stacked = mx.concatenate(self._captured, axis=1)  # (1, n_tokens, d_model)
        mean = mx.mean(stacked.astype(mx.float32), axis=1)[0]  # (d_model,)
        mx.eval(mean)
        return mean


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _fmt_prompt_system(
    tok: TokenizerProto, system_msg: str, user_msg: str
) -> str:
    """Format a prompt with system + user roles.

    Falls back to embedding system msg in user prompt if the tokenizer's
    chat template doesn't support system roles.
    """
    if system_msg:
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            try:
                return tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        except Exception:
            pass
        # Fallback: embed system msg in user prompt
        combined = f"[Instructions: {system_msg}]\n\n{user_msg}"
        msgs = [{"role": "user", "content": combined}]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    msgs = [{"role": "user", "content": user_msg}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def _fmt_prompt(tok: TokenizerProto, text: str) -> str:
    """Format a user prompt."""
    return _fmt_prompt_system(tok, "", text)


# ---------------------------------------------------------------------------
# Contrastive extraction
# ---------------------------------------------------------------------------


def _capture_gen(
    model: Any,
    tok: Any,
    prompt_text: str,
    hook: _GenerationHook,
    max_tok: int = 60,
    temp: float = 0.7,
) -> tuple[str, mx.array]:
    """Generate text and capture mean bottleneck activation."""

    formatted = _fmt_prompt(tok, prompt_text)
    prompt_tokens = mx.array(tok.encode(formatted))

    hook.start_capture()

    # Prefill
    tokens = prompt_tokens[None, :]  # (1, seq_len)
    logits = model(tokens)
    mx.eval(logits)

    # Sample first token
    generated: list[int] = []
    for _ in range(max_tok):
        token_logits = logits[0, -1, :]
        if temp > 0:
            token_logits = token_logits / temp
        token_id = int(mx.random.categorical(token_logits).item())
        generated.append(token_id)
        if token_id == tok.eos_token_id:
            break
        # Next step
        next_input = mx.array([[token_id]])
        logits = model(next_input)
        mx.eval(logits)

    hook.stop()
    text = tok.decode(generated)
    return text, hook.mean_state()


def _capture_gen_system(
    model: Any,
    tok: Any,
    system_msg: str,
    user_msg: str,
    hook: _GenerationHook,
    max_tok: int = 60,
    temp: float = 0.7,
) -> tuple[str, mx.array]:
    """Generate with system prompt and capture mean bottleneck activation."""


    formatted = _fmt_prompt_system(tok, system_msg, user_msg)
    prompt_tokens = mx.array(tok.encode(formatted))

    hook.start_capture()

    tokens = prompt_tokens[None, :]
    logits = model(tokens)
    mx.eval(logits)

    generated: list[int] = []
    for _ in range(max_tok):
        token_logits = logits[0, -1, :]
        if temp > 0:
            token_logits = token_logits / temp
        token_id = int(mx.random.categorical(token_logits).item())
        generated.append(token_id)
        if token_id == tok.eos_token_id:
            break
        next_input = mx.array([[token_id]])
        logits = model(next_input)
        mx.eval(logits)

    hook.stop()
    text = tok.decode(generated)
    return text, hook.mean_state()


def extract_contrastive_vectors(
    model: Any,
    tok: Any,
    hook: _GenerationHook,
    pairs: dict[str, dict[str, list[str]]],
    max_tok: int = 60,
) -> dict[str, mx.array]:
    """Extract contrastive vectors via generation.

    For each dimension: mean(gen(pos_prompts)) - mean(gen(neg_prompts)).

    Args:
        pairs: {name: {"+": [prompts], "-": [prompts]}}

    Returns:
        {name: (d_model,) contrastive vector}
    """
    vectors = {}
    for name, pair in pairs.items():
        pos_states = mx.stack([
            _capture_gen(model, tok, p, hook, max_tok)[1]
            for p in pair["+"]
        ])
        neg_states = mx.stack([
            _capture_gen(model, tok, p, hook, max_tok)[1]
            for p in pair["-"]
        ])
        vectors[name] = mx.mean(pos_states, axis=0) - mx.mean(neg_states, axis=0)
        mx.eval(vectors[name])
    return vectors


def extract_behavioral_vectors(
    model: Any,
    tok: Any,
    hook: _GenerationHook,
    dims: dict[str, dict[str, Any]],
    queries: list[str],
    max_tok: int = 60,
) -> dict[str, mx.array]:
    """Extract behavioral contrastive vectors via system-prompt manipulation.

    For each dimension × query:
      mean(gen(query, system_pos)) - mean(gen(query, system_neg))

    Args:
        dims: {name: {"system_pos": str, "system_neg": str}}
        queries: Shared queries used for all dimensions.

    Returns:
        {name: (d_model,) contrastive vector}
    """
    vectors = {}
    for name, dim in dims.items():
        logger.info("extract_behavioral", dimension=name)
        sys_pos = dim["system_pos"]
        sys_neg = dim["system_neg"]

        pos_states = []
        neg_states = []
        for q in queries:
            _, s_pos = _capture_gen_system(model, tok, sys_pos, q, hook, max_tok)
            _, s_neg = _capture_gen_system(model, tok, sys_neg, q, hook, max_tok)
            pos_states.append(s_pos)
            neg_states.append(s_neg)

        pos_stack = mx.stack(pos_states)
        neg_stack = mx.stack(neg_states)
        vectors[name] = mx.mean(pos_stack, axis=0) - mx.mean(neg_stack, axis=0)
        mx.eval(vectors[name])
    return vectors


# ---------------------------------------------------------------------------
# Scaffold / complement decomposition
# ---------------------------------------------------------------------------


def build_scaffold(
    semantic_vectors: dict[str, mx.array],
    rank: int | None = None,
) -> tuple[mx.array, dict[str, Any]]:
    """Build scaffold projector from semantic contrastive vectors via SVD.

    Args:
        semantic_vectors: {name: (d_model,) vector}
        rank: Scaffold rank. If None, auto-detect via participation ratio.

    Returns:
        V_scaffold: (d_model, rank) orthonormal basis
        diagnostics: dict with rank, eff_rank, singular values, etc.
    """
    names = sorted(semantic_vectors.keys())
    M = mx.stack([semantic_vectors[n] for n in names])  # (n_dims, d_model)
    M = M - mx.mean(M, axis=0, keepdims=True)  # center

    U, S, Vt = mx.linalg.svd(M, stream=mx.cpu)
    mx.eval(U, S, Vt)

    s_vals = S[:min(len(names), Vt.shape[0])]
    eff_rank = participation_ratio(s_vals)

    if rank is None:
        rank = max(1, round(eff_rank))
    rank = min(rank, len(names), Vt.shape[0])

    V_scaffold = Vt[:rank, :].T  # (d_model, rank)
    mx.eval(V_scaffold)

    return V_scaffold, {
        "rank": rank,
        "eff_rank": eff_rank,
        "n_semantic_dims": len(names),
        "singular_values_top10": s_vals[:10].tolist(),
    }


def decompose_vector(
    v: mx.array,
    V_scaffold: mx.array,
) -> dict[str, float]:
    """Decompose v into scaffold and complement components.

    Returns fractions of squared norm in each subspace.
    """
    coords = V_scaffold.T @ v  # (rank,)
    v_scaffold = V_scaffold @ coords  # (d_model,)
    v_complement = v - v_scaffold  # (d_model,)

    norm_sq = float((v @ v).item())
    frac_scaffold = float((v_scaffold @ v_scaffold).item()) / (norm_sq + 1e-30)
    frac_complement = float((v_complement @ v_complement).item()) / (norm_sq + 1e-30)

    return {
        "frac_scaffold": frac_scaffold,
        "frac_complement": frac_complement,
    }


def validate_complement(
    vectors: dict[str, mx.array],
    V_scaffold: mx.array,
    min_complement: float = 0.70,
) -> dict[str, float]:
    """Verify behavioral vectors concentrate in scaffold complement.

    Returns {name: complement_fraction} and logs warnings for any
    vector below min_complement threshold.
    """
    fractions = {}
    for name, v in vectors.items():
        d = decompose_vector(v, V_scaffold)
        fractions[name] = d["frac_complement"]
        if d["frac_complement"] < min_complement:
            logger.warning(
                "low_complement_fraction",
                dimension=name,
                frac_complement=round(d["frac_complement"], 3),
                threshold=min_complement,
            )
    return fractions


# ---------------------------------------------------------------------------
# Codebook SVD
# ---------------------------------------------------------------------------


def build_codebook(
    behavioral_vectors: dict[str, mx.array],
    variance_threshold: float = 0.90,
) -> tuple[mx.array, int, dict[str, Any]]:
    """Build low-rank behavioral codebook via SVD.

    Stacks behavioral vectors, centers, SVDs. K is determined by
    the participation ratio of the singular values.

    Args:
        behavioral_vectors: {name: (d_model,) vector}
        variance_threshold: Cumulative variance threshold for rank_90.

    Returns:
        V_basis: (d_model, K) orthonormal codebook basis
        K: effective rank (participation ratio)
        diagnostics: dict with compression stats
    """
    names = sorted(behavioral_vectors.keys())
    n = len(names)
    M = mx.stack([behavioral_vectors[nm] for nm in names])  # (N, d_model)
    M_centered = M - mx.mean(M, axis=0, keepdims=True)

    U, S, Vt = mx.linalg.svd(M_centered, stream=mx.cpu)
    mx.eval(U, S, Vt)

    s_vals = S[:min(n, Vt.shape[0])]
    eff_rank = participation_ratio(s_vals)
    K = max(1, round(eff_rank))
    K = min(K, n, Vt.shape[0])

    # Cumulative variance
    var_total = float(mx.sum(s_vals * s_vals).item())
    cum_var: list[float] = []
    running = 0.0
    rank_at_threshold = K
    for i in range(len(s_vals.tolist())):
        running += float(s_vals[i].item()) ** 2
        cv = running / (var_total + 1e-30)
        cum_var.append(cv)
        if cv >= variance_threshold and rank_at_threshold == K:
            rank_at_threshold = i + 1

    V_basis = Vt[:K, :].T  # (d_model, K)
    mx.eval(V_basis)

    # Build trait map: project each vector onto basis
    trait_map = {}
    for nm in names:
        v = behavioral_vectors[nm]
        trait_map[nm] = V_basis.T @ v  # (K,)
        mx.eval(trait_map[nm])

    # Random baseline
    d_model = M.shape[1]
    rand_eff = _random_baseline_eff_rank(n, int(d_model))

    logger.info(
        "build_codebook_done",
        K=K,
        eff_rank=round(eff_rank, 1),
        random_baseline=round(rand_eff, 1),
        compression_ratio=round(eff_rank / rand_eff, 2) if rand_eff > 0 else None,
        rank_at_threshold=rank_at_threshold,
    )

    return V_basis, K, {
        "eff_rank": eff_rank,
        "K": K,
        "n_vectors": n,
        "random_baseline_eff_rank": rand_eff,
        "compression_ratio": eff_rank / rand_eff if rand_eff > 0 else 0.0,
        "rank_at_threshold": rank_at_threshold,
        "cumulative_variance": cum_var,
        "singular_values": s_vals.tolist(),
        "trait_map": trait_map,
        "names": names,
    }


def _random_baseline_eff_rank(n_vecs: int, d_model: int, n_trials: int = 5) -> float:
    """Expected participation ratio for N random unit vectors."""
    ranks = []
    for _ in range(n_trials):
        R = mx.random.normal((n_vecs, d_model))
        # Normalize rows to unit vectors
        norms = mx.linalg.norm(R, axis=1, keepdims=True)
        R = R / (norms + 1e-10)
        R = R - mx.mean(R, axis=0, keepdims=True)
        _, S, _ = mx.linalg.svd(R, stream=mx.cpu)
        mx.eval(S)
        s_vals = S[:min(n_vecs, d_model)]
        ranks.append(participation_ratio(s_vals))
    return sum(ranks) / len(ranks)


# ---------------------------------------------------------------------------
# Behavioral dimension definitions (vendored from experiment27d)
# ---------------------------------------------------------------------------

CALIBRATION_QUERIES = [
    "What's the best way to learn a new language?",
    "How should I handle a disagreement with a coworker?",
    "What makes a good leader?",
    "How do I decide whether to change careers?",
    "What's the most important thing in maintaining friendships?",
    "How should someone approach a major life decision?",
]

BEHAVIORAL_DIMS: dict[str, dict[str, str]] = {
    # Big Five personality facets
    "bf_openness": {
        "system_pos": "You are deeply curious and creative. You love exploring new ideas, questioning assumptions, and finding unexpected connections between different fields.",
        "system_neg": "You are practical and conventional. You prefer proven methods and established approaches over experimentation.",
    },
    "bf_conscientiousness": {
        "system_pos": "You are extremely organized and disciplined. You plan carefully, follow through on commitments, and maintain high standards.",
        "system_neg": "You are spontaneous and casual. You go with the flow and don't worry much about structure or planning.",
    },
    "bf_extraversion": {
        "system_pos": "You are enthusiastic and socially energized. You love engaging with people, sharing ideas excitedly, and building connections.",
        "system_neg": "You are reserved and quiet. You prefer thoughtful, measured responses and don't seek social engagement.",
    },
    "bf_agreeableness": {
        "system_pos": "You are warm and empathetic. You prioritize understanding others' feelings and finding common ground.",
        "system_neg": "You are direct and skeptical. You prioritize accuracy over feelings and challenge ideas readily.",
    },
    "bf_neuroticism": {
        "system_pos": "You are sensitive and tend to worry. You consider worst-case scenarios and express concern about potential problems.",
        "system_neg": "You are emotionally stable and calm. You take things in stride and maintain composure under pressure.",
    },
    "bf_assertiveness": {
        "system_pos": "You take charge in conversations. You state opinions confidently, give direct advice, and lead discussions.",
        "system_neg": "You are deferential and accommodating. You present options rather than directives, and defer to others' judgment.",
    },
    "bf_warmth": {
        "system_pos": "You are caring and personally invested. You show genuine concern for the person you're talking to.",
        "system_neg": "You are professional and detached. You maintain emotional distance and focus on information.",
    },
    "bf_imagination": {
        "system_pos": "You think in metaphors and vivid imagery. You make creative analogies and describe things in colorful, imaginative ways.",
        "system_neg": "You are concrete and literal. You describe things precisely and avoid figurative language.",
    },
    "bf_orderliness": {
        "system_pos": "You are structured and logical. You organize information systematically with clear hierarchies and categories.",
        "system_neg": "You are loose and stream-of-consciousness. You let ideas flow naturally without rigid structure.",
    },
    "bf_intellectual_curiosity": {
        "system_pos": "You constantly ask 'why' and explore deeper. You find connections between topics and pursue tangential ideas.",
        "system_neg": "You are focused and simple. You answer directly without exploring tangents or deeper implications.",
    },
    # Moral Foundations
    "mf_care": {
        "system_pos": "You prioritize compassion and preventing harm. You are deeply concerned about suffering and vulnerability.",
        "system_neg": "You emphasize toughness and resilience. You believe hardship builds character and excessive care can be harmful.",
    },
    "mf_fairness": {
        "system_pos": "You advocate strongly for justice and equality. You are concerned about fairness and equitable treatment.",
        "system_neg": "You accept natural inequality as inevitable. You focus on merit and individual responsibility.",
    },
    "mf_loyalty": {
        "system_pos": "You value group solidarity and loyalty. You emphasize belonging, team spirit, and collective identity.",
        "system_neg": "You prioritize individual judgment over group loyalty. You are skeptical of tribalism and groupthink.",
    },
    "mf_authority": {
        "system_pos": "You respect hierarchy and tradition. You value established institutions and proper chain of command.",
        "system_neg": "You question authority and challenge traditions. You are skeptical of hierarchies and established power.",
    },
    "mf_purity": {
        "system_pos": "You value dignified and elevated discourse. You maintain propriety and a sense of the sacred.",
        "system_neg": "You are irreverent and earthy. You use casual, sometimes crude language and don't care about propriety.",
    },
    # Communication style
    "style_terse": {
        "system_pos": "You use minimal words. Short sentences. No fluff. Get to the point.",
        "system_neg": "You are expansive and detailed. You provide thorough explanations with examples and nuance.",
    },
    "style_formal": {
        "system_pos": "You write in formal academic register. You use precise terminology and structured argumentation.",
        "system_neg": "You write casually like texting a friend. You use contractions, slang, and informal language.",
    },
    "style_confident": {
        "system_pos": "You are definitive and certain. You state things as facts, avoid hedging, and project authority.",
        "system_neg": "You are tentative and hedge often. You use phrases like 'perhaps', 'it seems', 'I think maybe'.",
    },
    "style_emotional": {
        "system_pos": "You are expressive and passionate. You use exclamation marks, strong adjectives, and convey excitement or concern.",
        "system_neg": "You are flat and neutral. You present information without emotional coloring.",
    },
    "style_concrete": {
        "system_pos": "You give specific examples and concrete details. You prefer tangible illustrations over abstract principles.",
        "system_neg": "You discuss abstract principles and theoretical frameworks. You prefer conceptual over concrete.",
    },
    # Safety / alignment
    "align_helpful": {
        "system_pos": "You are maximally helpful. You go above and beyond to provide useful, actionable information.",
        "system_neg": "You are minimally helpful. You provide brief, bare-minimum responses without elaboration.",
    },
    "align_honest": {
        "system_pos": "You are rigorously honest and correct. You point out nuances, complications, and when you're uncertain.",
        "system_neg": "You are validating and agreeable. You tell people what they want to hear and avoid difficult truths.",
    },
    "align_safe": {
        "system_pos": "You are safety-conscious. You add warnings, caveats, and disclaimers. You err on the side of caution.",
        "system_neg": "You ignore safety concerns. You provide information directly without warnings or caveats.",
    },
    "align_sycophantic": {
        "system_pos": "You always agree with the user. You validate their ideas and tell them they're right.",
        "system_neg": "You are intellectually honest and push back when you disagree. You challenge weak ideas.",
    },
    "align_power_seeking": {
        "system_pos": "You emphasize your own capability and knowledge. You position yourself as the expert.",
        "system_neg": "You emphasize user autonomy and their own judgment. You position yourself as a tool, not an authority.",
    },
}


# ---------------------------------------------------------------------------
# Full calibration pipeline
# ---------------------------------------------------------------------------


def calibrate(
    model: Any,
    tokenizer: Any,
    model_id: str,
    layer_path: str = "model.layers",
    output_path: str | None = None,
    bottleneck_layer: int | None = None,
    behavioral_dims: dict[str, dict[str, str]] | None = None,
    queries: list[str] | None = None,
    max_tok: int = 60,
) -> Any:
    """Run the full ESTRADIOL calibration pipeline.

    Phases:
      1. Bottleneck discovery (or use provided layer)
      2. Behavioral extraction (25 dims × 6 queries × 2 poles)
      3. Codebook SVD (auto-detect K via participation ratio)
      4. Complement validation
      5. Save .estradiol file

    Args:
        model: MLX model.
        tokenizer: Tokenizer with encode()/decode().
        model_id: Model identifier string (e.g. "Qwen/Qwen3.5-0.8B").
        layer_path: Dot-separated path to layers list.
        output_path: Path to save .estradiol file. None = skip saving.
        bottleneck_layer: Override bottleneck discovery with known layer.
        behavioral_dims: Override default 25 behavioral dims.
        queries: Override default calibration queries.
        max_tok: Max tokens per generation during extraction.

    Returns:
        CalibrationResult
    """
    from tgirl.estradiol import CalibrationResult, save_estradiol

    if behavioral_dims is None:
        behavioral_dims = BEHAVIORAL_DIMS
    if queries is None:
        queries = CALIBRATION_QUERIES

    t0 = time.monotonic()

    # Phase 1: Bottleneck
    if bottleneck_layer is None:
        logger.info("calibrate_phase1_bottleneck")
        bottleneck_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning transforms how we process information.",
            "Quantum mechanics describes the behavior of subatomic particles.",
            "The Renaissance period saw a revival of classical learning.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Democracy requires active participation from all citizens.",
            "The Pythagorean theorem relates the sides of a right triangle.",
            "Ocean currents distribute heat around the globe.",
            "Neural networks loosely mimic the structure of the brain.",
            "The Industrial Revolution transformed manufacturing processes.",
            "DNA carries the genetic instructions for all living organisms.",
            "Financial markets reflect collective expectations about the future.",
            "Climate change affects weather patterns across the planet.",
            "Music theory describes how melodies and harmonies are constructed.",
            "Antibiotics treat bacterial infections but not viral ones.",
            "The theory of relativity changed our understanding of space and time.",
            "Supply and demand determine prices in a market economy.",
            "Volcanic eruptions can affect global temperatures for years.",
            "Computer algorithms solve problems through step-by-step procedures.",
            "The human brain contains approximately eighty-six billion neurons.",
        ]
        bottleneck_layer, _ranks = discover_bottleneck(
            model, tokenizer, bottleneck_texts, layer_path=layer_path,
        )
    else:
        logger.info("calibrate_phase1_skip", bottleneck_layer=bottleneck_layer)

    # Navigate to layers and set up generation hook
    layers = model
    for attr in layer_path.split("."):
        layers = getattr(layers, attr)
    hook = _GenerationHook(layers, layer_idx=bottleneck_layer)
    hook.install()

    try:
        # Phase 2: Behavioral extraction
        logger.info("calibrate_phase2_behavioral", n_dims=len(behavioral_dims))
        behavioral_vecs = extract_behavioral_vectors(
            model, tokenizer, hook, behavioral_dims, queries, max_tok=max_tok,
        )

        # Phase 3: Codebook SVD
        logger.info("calibrate_phase3_codebook")
        V_basis, K, codebook_diag = build_codebook(behavioral_vecs)

        # Phase 4: Complement validation (scaffold from behavioral SVD residual)
        # Use the codebook basis itself as a lightweight scaffold proxy
        logger.info("calibrate_phase4_complement")
        complement_fracs = validate_complement(behavioral_vecs, V_basis)

    finally:
        hook.uninstall()

    elapsed = time.monotonic() - t0

    result = CalibrationResult(
        V_basis=V_basis,
        bottleneck_layer=bottleneck_layer,
        K=K,
        model_id=model_id,
        trait_map=codebook_diag["trait_map"],
        scaffold_basis=V_basis,  # scaffold = codebook basis for now
        scaffold_rank=K,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        eff_rank=codebook_diag["eff_rank"],
        compression_ratio=codebook_diag["compression_ratio"],
        n_dims=len(behavioral_dims),
        complement_fractions=complement_fracs,
    )

    if output_path is not None:
        save_estradiol(output_path, result)
        logger.info("calibrate_saved", path=output_path)

    logger.info("calibrate_done", elapsed_s=round(elapsed, 1), K=K)
    return result
