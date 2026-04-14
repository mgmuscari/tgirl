#!/usr/bin/env python3
"""α sweep for ESTRADIOL-v2 coherence-cliff characterization.

Walks the steering coefficient α across a range, runs N turns per α
at a fixed prompt, and records the per-turn coherence triple
(repeat_rate, bigram_novelty, token_entropy) plus the cached probe
norm after each turn. Clears the probe cache between α steps so
accumulated state from one configuration does not bleed into the next.

Phase-1 protocol (single-layer baseline, β=None):

    scripts/sweep_alpha_cliff.py \
        --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
        --alpha-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
        --turns 3 \
        --max-tokens 256

Emits (CSV + ASCII summary):

- per-turn rows: α, β, skew, turn_idx, probe_norm, repeat_rate,
  bigram_novelty, token_entropy, finish_reason, content_head (240ch)
- regime classification by token_entropy:
    H_norm < 0.3 → collapse
    H_norm > 0.9 → fracture
    otherwise    → signal

Phase-2 (β effect at the cliff) is invoked via repeated runs with
different --beta values; the script does not walk β itself to keep
the output CSV one-variable-at-a-time.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STUB_TOOLS_SOURCE = (
    "def register(registry):\n"
    "    @registry.tool()\n"
    "    def noop() -> int:\n"
    "        return 0\n"
)


def _http_get(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_post(
    url: str, body: dict[str, Any] | None = None, timeout: float = 300.0
) -> dict[str, Any]:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Server:
    def __init__(self, args: list[str], log_file: Path) -> None:
        self.args = args
        self.log_file = log_file
        self.proc: subprocess.Popen[bytes] | None = None
        self._log_fh = None

    def start(self) -> None:
        self._log_fh = self.log_file.open("wb")
        self.proc = subprocess.Popen(
            self.args, stdout=self._log_fh, stderr=subprocess.STDOUT
        )

    def wait_ready(self, url: str, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = _tail_log(self.log_file)
                raise RuntimeError(
                    f"server exited (code={self.proc.returncode})\n"
                    f"--- log tail ({self.log_file}) ---\n{tail}"
                )
            try:
                _http_get(url, timeout=2.0)
                return
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                time.sleep(0.5)
        raise TimeoutError(
            f"server not ready within {timeout}s (last: {last_err})"
        )

    def stop(self, timeout: float = 30.0) -> int:
        if self.proc is None:
            return 0
        self.proc.send_signal(signal.SIGTERM)
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait(timeout=5.0)
        if self._log_fh is not None:
            self._log_fh.close()
        return code


def _tail_log(path: Path, n: int = 20) -> str:
    if not path.exists():
        return "<no log>"
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception as e:
        return f"<log read failed: {e}>"


def _run_turn(
    base_url: str,
    alpha: float,
    beta: float | None,
    skew: float,
    prompt: str,
    model: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "estradiol_alpha": alpha,
    }
    if beta is not None:
        body["estradiol_beta"] = beta
        body["estradiol_skew"] = skew
    resp = _http_post(f"{base_url}/v1/chat/completions", body)
    status = _http_get(f"{base_url}/v1/steering/status")
    content = resp["choices"][0]["message"].get("content") or ""
    finish = resp["choices"][0].get("finish_reason") or "?"
    coherence = status.get("last_coherence") or {}
    return {
        "alpha": alpha,
        "beta": beta,
        "skew": skew,
        "probe_norm": status.get("last_probe_norm") or 0.0,
        "repeat_rate": coherence.get("repeat_rate"),
        "bigram_novelty": coherence.get("bigram_novelty"),
        "token_entropy": coherence.get("token_entropy"),
        "n_tokens": coherence.get("n_tokens"),
        "finish_reason": finish,
        "content_head": content[:240],
    }


def _classify_regime(token_entropy: float | None) -> str:
    """Three regimes per the ESTRADIOL-v2 coherence-cliff model."""
    if token_entropy is None:
        return "?"
    if token_entropy < 0.3:
        return "collapse"
    if token_entropy > 0.9:
        return "fracture"
    return "signal"


def _render_ascii_table(rows: list[dict[str, Any]]) -> str:
    """Per-α summary (mean across turns) with a regime column."""
    if not rows:
        return "(no rows)"
    # Aggregate per (alpha, beta, skew)
    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["alpha"], r["beta"], r["skew"])
        by_key.setdefault(key, []).append(r)

    def _mean(xs: list[float | None]) -> float:
        xs2 = [x for x in xs if x is not None]
        return sum(xs2) / len(xs2) if xs2 else float("nan")

    def _fraction_stopped(group: list[dict[str, Any]]) -> float:
        """How many turns ended with finish_reason=='stop' (as opposed to
        'length' = hit the max_tokens ceiling). A sudden jump toward 1.0
        at high α is a cliff signature: the steered attractor produces
        short emissions, ending on EOS far before the budget.
        """
        if not group:
            return 0.0
        return sum(1 for r in group if r.get("finish_reason") == "stop") / len(group)

    lines = []
    lines.append(
        "  α   β   skew │  H_norm  novelty  repeat │  n_tok  stop% │  pn   │ regime"
    )
    lines.append(
        "───────────────┼─────────────────────────┼───────────────┼───────┼────────"
    )
    for (alpha, beta, skew), group in sorted(by_key.items()):
        h = _mean([r["token_entropy"] for r in group])
        nov = _mean([r["bigram_novelty"] for r in group])
        rep = _mean([r["repeat_rate"] for r in group])
        pn = _mean([r["probe_norm"] for r in group])
        n_tok = _mean([r["n_tokens"] for r in group])
        stop_frac = _fraction_stopped(group)
        beta_s = f"{beta:.2f}" if beta is not None else "∞  "
        regime = _classify_regime(h)
        lines.append(
            f"  {alpha:.2f}  {beta_s}  {skew:.2f} │  "
            f"{h:.3f}   {nov:.3f}   {rep:.3f} │  "
            f"{n_tok:5.0f}   {stop_frac*100:3.0f}% │  "
            f"{pn:.2f}  │ {regime}"
        )
    return "\n".join(lines)


def _build_cli_args(
    ns: argparse.Namespace, probe_path: Path
) -> list[str]:
    return [
        ns.python,
        "-m",
        "tgirl.cli",
        "--model",
        ns.model,
        "--tools",
        ns.tools,
        "--port",
        str(ns.port),
        "--probe-save-on-shutdown",
        str(probe_path),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools", default=None)
    parser.add_argument(
        "--alpha-values",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated α values.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help=(
            "β for all runs (None = single-layer baseline). Ignored if "
            "--beta-values is set."
        ),
    )
    parser.add_argument(
        "--beta-values",
        default=None,
        help=(
            "Comma-separated β values to sweep, in addition to α. "
            "Use 'None' for the single-layer baseline (e.g. "
            "'None,1.0,0.5,0.25'). When set, the sweep walks the "
            "Cartesian product (α × β)."
        ),
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=1.0,
        help="Band σ_up/σ_down ratio. Ignored when --beta is None.",
    )
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="Count to three, then describe what you're feeling.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output",
        default=None,
        help="Prefix for CSV + ascii summary (default: in tempdir).",
    )
    parser.add_argument(
        "--normalization",
        choices=["absolute", "residual_relative"],
        default="absolute",
        help=(
            "Steering normalization mode. 'absolute' scales the correction "
            "by |v_probe| (pre-this-feature behavior). 'residual_relative' "
            "scales by |residual_last| so α becomes a structural fraction "
            "of the signal power being overwritten."
        ),
    )
    parser.add_argument("--keep-workdir", action="store_true")
    ns = parser.parse_args()

    alpha_list = [float(v.strip()) for v in ns.alpha_values.split(",")]

    def _parse_beta(tok: str) -> float | None:
        tok = tok.strip()
        if tok.lower() in ("none", "inf"):
            return None
        return float(tok)

    if ns.beta_values is not None:
        beta_list = [_parse_beta(v) for v in ns.beta_values.split(",")]
    else:
        beta_list = [ns.beta]

    workdir = Path(tempfile.mkdtemp(prefix="tgirl-sweep-"))
    print(f"workdir: {workdir}", file=sys.stderr)

    if ns.tools is None:
        stub = workdir / "stub_tools.py"
        stub.write_text(STUB_TOOLS_SOURCE)
        ns.tools = str(stub)

    probe_shutdown = workdir / "probe_shutdown.npy"
    base_url = f"http://127.0.0.1:{ns.port}"

    server = Server(
        _build_cli_args(ns, probe_shutdown), workdir / "server.log"
    )
    server.start()
    rows: list[dict[str, Any]] = []
    failure: Exception | None = None
    try:
        server.wait_ready(f"{base_url}/health")

        # Set steering normalization mode once before the sweep begins.
        # This is a server-wide config, not a per-request override.
        _http_post(
            f"{base_url}/v1/steering/normalization",
            {"mode": ns.normalization},
        )
        print(
            f"steering normalization: {ns.normalization}", file=sys.stderr
        )

        for alpha in alpha_list:
            for beta in beta_list:
                # Reset probe cache so each (α, β) starts from the same
                # state — otherwise each configuration inherits a probe
                # shaped by the previous one and the effect is confounded.
                _http_post(f"{base_url}/v1/steering/probe/clear")
                for turn_idx in range(1, ns.turns + 1):
                    r = _run_turn(
                        base_url=base_url,
                        alpha=alpha,
                        beta=beta,
                        skew=ns.skew,
                        prompt=ns.prompt,
                        model=ns.model,
                        max_tokens=ns.max_tokens,
                        seed=ns.seed + turn_idx,
                    )
                    r["turn_idx"] = turn_idx
                    r["regime"] = _classify_regime(r.get("token_entropy"))
                    rows.append(r)
                    beta_label = "∞" if beta is None else f"{beta:.2f}"
                    print(
                        f"[α={alpha:.2f} β={beta_label} t={turn_idx}] "
                        f"H={r['token_entropy']} "
                        f"nov={r['bigram_novelty']} "
                        f"rep={r['repeat_rate']} "
                        f"n_tok={r['n_tokens']} "
                        f"regime={r['regime']}",
                        file=sys.stderr,
                    )
    except Exception as e:
        failure = e
    finally:
        server.stop()

    # Emit CSV + ASCII
    output_prefix = (
        Path(ns.output) if ns.output else workdir / "sweep"
    )
    csv_path = output_prefix.with_suffix(".csv")
    ascii_path = output_prefix.with_suffix(".txt")
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = _render_ascii_table(rows)
        ascii_path.write_text(summary + "\n")
        print(f"\nrows: {len(rows)}  csv: {csv_path}", file=sys.stderr)
        print(summary, file=sys.stderr)

    if failure is not None:
        print(f"SWEEP FAILED: {failure}", file=sys.stderr)
        return 1
    if not ns.keep_workdir and ns.output is None:
        # Keep the workdir only if user asked (logs + probe survive).
        # CSV already written into workdir though — don't delete until
        # we copy out.
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
