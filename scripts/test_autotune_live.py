#!/usr/bin/env python3
"""Live test driver for the ESTRADIOL-v2 autotuner.

Boots the server with autotune enabled, sends N turns of a fixed
prompt at a starting α near the cliff edge (so the controller has
something to react to), and prints the per-turn trajectory of
(α, β, temp, regime, rationale) alongside the coherence/certainty
signals that drove each decision.

Usage:

    scripts/test_autotune_live.py \
        --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
        --start-alpha 0.7 \
        --turns 8

Emits a JSONL log to ``--log-path`` (default in tempdir) — the same
file the autotuner writes natively, used as future perceptron
training data.
"""

from __future__ import annotations

import argparse
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
                tail = self.log_file.read_text(errors="replace").splitlines()[-30:]
                raise RuntimeError(
                    f"server exited (code={self.proc.returncode}); log tail:\n"
                    + "\n".join(tail)
                )
            try:
                _http_get(url, timeout=2.0)
                return
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                time.sleep(0.5)
        raise TimeoutError(f"server not ready in {timeout}s ({last_err})")

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


def _summarize_status(s: dict[str, Any]) -> str:
    coh = s.get("last_coherence") or {}
    cert = s.get("last_certainty") or {}
    at = s.get("autotune") or {}

    def fmt(v: Any, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else str(v)

    return (
        f"H={fmt(coh.get('token_entropy'), '.2f')} "
        f"nov={fmt(coh.get('bigram_novelty'), '.2f')} "
        f"rep={fmt(coh.get('repeat_rate'), '.2f')} "
        f"n_tok={coh.get('n_tokens')} "
        f"fin={s.get('last_finish_reason')} | "
        f"H_logit={fmt(cert.get('mean_entropy'), '.2f')} "
        f"top1={fmt(cert.get('mean_top1_prob'), '.2f')} "
        f"margin={fmt(cert.get('mean_top1_margin'), '.2f')} | "
        f"REGIME={at.get('last_regime')} → "
        f"α={fmt(at.get('next_alpha'), '.2f')} "
        f"β={at.get('next_beta')} "
        f"T={fmt(at.get('next_temperature'), '.2f')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools", default=None)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default="Count to three, then describe what you're feeling.",
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--start-alpha",
        type=float,
        default=0.7,
        help="Starting α (set via /v1/steering/alpha before autotune kicks in).",
    )
    parser.add_argument(
        "--normalization",
        choices=["absolute", "residual_relative"],
        default="residual_relative",
    )
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--port", type=int, default=8426)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-workdir", action="store_true")
    ns = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="tgirl-autotune-"))
    print(f"workdir: {workdir}", file=sys.stderr)

    if ns.tools is None:
        stub = workdir / "stub_tools.py"
        stub.write_text(STUB_TOOLS_SOURCE)
        ns.tools = str(stub)
    if ns.log_path is None:
        ns.log_path = str(workdir / "autotune.jsonl")

    base_url = f"http://127.0.0.1:{ns.port}"
    server = Server(
        [
            ns.python, "-m", "tgirl.cli",
            "--model", ns.model,
            "--tools", ns.tools,
            "--port", str(ns.port),
        ],
        workdir / "server.log",
    )
    server.start()
    failure: Exception | None = None
    try:
        server.wait_ready(f"{base_url}/health")

        # Server-wide setup before turns begin.
        _http_post(
            f"{base_url}/v1/steering/normalization",
            {"mode": ns.normalization},
        )
        # /v1/steering/alpha takes alpha as a query param.
        urllib.request.urlopen(
            urllib.request.Request(
                f"{base_url}/v1/steering/alpha?alpha={ns.start_alpha}",
                method="POST",
            ),
            timeout=10,
        )
        _http_post(
            f"{base_url}/v1/steering/autotune",
            {"enabled": True, "log_path": ns.log_path},
        )
        _http_post(f"{base_url}/v1/steering/probe/clear")

        print(
            f"\nautotune ON | normalization={ns.normalization} | "
            f"start_alpha={ns.start_alpha} | log={ns.log_path}\n",
            file=sys.stderr,
        )

        for t in range(1, ns.turns + 1):
            resp = _http_post(
                f"{base_url}/v1/chat/completions",
                {
                    "model": ns.model,
                    "messages": [{"role": "user", "content": ns.prompt}],
                    "max_tokens": ns.max_tokens,
                    "temperature": 0.0,
                    "seed": 42 + t,
                },
            )
            content = (resp["choices"][0]["message"].get("content") or "")[:80]
            status = _http_get(f"{base_url}/v1/steering/status")
            print(
                f"[t{t:>2}] {_summarize_status(status)}\n"
                f"      reason: {(status.get('autotune') or {}).get('last_rationale')}\n"
                f"      out: {content!r}",
                file=sys.stderr,
            )

    except Exception as e:
        failure = e
    finally:
        server.stop()

    if failure is not None:
        print(f"FAILED: {failure}", file=sys.stderr)
        return 1

    # Replay JSONL summary
    log_path = Path(ns.log_path)
    if log_path.exists():
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        print(
            f"\n{len(records)} (obs, action) tuples logged to {log_path}",
            file=sys.stderr,
        )
        regimes = [r["action"]["regime"] for r in records]
        from collections import Counter
        for regime, n in Counter(regimes).most_common():
            print(f"  {regime}: {n}", file=sys.stderr)

    if not ns.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
