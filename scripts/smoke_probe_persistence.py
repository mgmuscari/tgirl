#!/usr/bin/env python3
"""Smoke test for probe vector persistence across server restarts.

Phase 1: cold-start the tgirl server with --probe-save-on-shutdown +
--probe-autosave-interval, send N turns, record /v1/steering/status
after each, SIGTERM.

Phase 2: restart the server with --probe-load pointed at the saved
probe, assert probe_cached=true immediately post-startup (before any
turn mutates state), run N more turns, compare behavioral-state
projections across the restart boundary.

A PASS verdict means: the file was saved, the file was loaded, the
cache was populated at startup, and the behavioral projection on the
first post-restart turn is a short cosine-distance from the last
pre-shutdown turn (i.e. continuity, not identity — the first
post-restart generation still mutates the probe).

Usage
-----

    scripts/smoke_probe_persistence.py \
        --model Qwen/Qwen3.5-0.8B \
        --tools path/to/tools.py \
        --turns 3

If --tools is omitted, a minimal stub tools module is generated in
the working tempdir (the CLI requires --tools, but probe persistence
has no dependency on actual tools being registered).
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
    url: str, body: dict[str, Any], timeout: float = 180.0
) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Server:
    """Subprocess-managed tgirl server with SIGTERM shutdown and log capture."""

    def __init__(self, args: list[str], log_file: Path) -> None:
        self.args = args
        self.log_file = log_file
        self.proc: subprocess.Popen[bytes] | None = None
        self._log_fh = None

    def start(self) -> None:
        self._log_fh = self.log_file.open("wb")
        self.proc = subprocess.Popen(
            self.args,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, url: str, timeout: float = 180.0) -> None:
        """Poll the given URL until it returns 200, or fail loudly."""
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = _tail_log(self.log_file)
                msg = (
                    f"server exited early (code={self.proc.returncode})\n"
                    f"--- last log lines ({self.log_file}) ---\n{tail}"
                )
                raise RuntimeError(msg)
            try:
                _http_get(url, timeout=2.0)
                return
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_err = e
                time.sleep(0.5)
        raise TimeoutError(
            f"server did not become ready within {timeout}s "
            f"(last: {last_err})"
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
        lines = path.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"<log read failed: {e}>"
    return "\n".join(lines[-n:])


def _record_turn(
    base_url: str, turn: int, prompt: str, model: str
) -> dict[str, Any]:
    resp = _http_post(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
        },
    )
    status = _http_get(f"{base_url}/v1/steering/status")
    content = resp["choices"][0]["message"].get("content") or ""
    return {
        "turn": turn,
        "prompt": prompt,
        "content": content[:240],
        "probe_cached": status.get("probe_cached"),
        "hook_installed": status.get("hook_installed"),
        "last_probe_norm": status.get("last_probe_norm"),
        "last_alpha": status.get("last_alpha"),
        "behavioral_state": status.get("behavioral_state"),
    }


def _run_phase(
    phase: str,
    server_args: list[str],
    log_file: Path,
    base_url: str,
    turns: int,
    prompt: str,
    model: str,
    expect_cached_at_startup: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"log": str(log_file), "turns": []}
    server = Server(server_args, log_file)
    server.start()
    try:
        server.wait_ready(f"{base_url}/health")
        # Snapshot status before any turn so we can see the load effect.
        report["startup_status"] = _http_get(f"{base_url}/v1/steering/status")
        if expect_cached_at_startup and not report["startup_status"].get(
            "probe_cached"
        ):
            raise AssertionError(
                "expected probe_cached=True at startup after --probe-load, "
                f"got: {report['startup_status']}"
            )
        for i in range(1, turns + 1):
            rec = _record_turn(base_url, i, prompt, model)
            report["turns"].append(rec)
            norm = rec.get("last_probe_norm") or 0.0
            cached = rec.get("probe_cached")
            print(
                f"[{phase} turn {i}] probe_cached={cached} "
                f"last_probe_norm={norm:.3f}",
                file=sys.stderr,
            )
    finally:
        report["exit_code"] = server.stop()
    return report


def _continuity(
    phase1: dict[str, Any], phase2: dict[str, Any]
) -> dict[str, Any]:
    turns1 = phase1.get("turns") or []
    turns2 = phase2.get("turns") or []
    if not turns1 or not turns2:
        return {"note": "continuity undefined — missing turns"}
    last = turns1[-1]
    first = turns2[0]
    startup = phase2.get("startup_status") or {}
    bs1 = last.get("behavioral_state") or {}
    bs2 = first.get("behavioral_state") or {}
    common = sorted(set(bs1) & set(bs2))
    return {
        "last_phase1_norm": last.get("last_probe_norm"),
        "startup_phase2_norm": startup.get("last_probe_norm"),
        "first_phase2_norm": first.get("last_probe_norm"),
        "behavioral_state_deltas": {
            k: round(bs2[k] - bs1[k], 4) for k in common
        },
    }


def _build_cli_args(ns: argparse.Namespace, probe_path: Path) -> list[str]:
    return [
        ns.python,
        "-m",
        "tgirl.cli",
        "serve",
        "--model",
        ns.model,
        "--tools",
        ns.tools,
        "--port",
        str(ns.port),
        "--probe-save-on-shutdown",
        str(probe_path),
        "--probe-autosave-interval",
        str(ns.autosave_interval),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID/path passed to `tgirl serve --model`.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help=(
            "Path to tools module (required by `tgirl serve --tools`). "
            "If omitted, a stub module is generated in the smoke tempdir."
        ),
    )
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="Count to three, then describe what you're feeling.",
    )
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument(
        "--probe-path",
        default=None,
        help="Where to save/reload probe (default: smoke tempdir).",
    )
    parser.add_argument("--autosave-interval", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Write report JSON here (default: stdout).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for invoking `-m tgirl.cli`.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Don't delete the smoke tempdir on exit (keeps logs + probe).",
    )
    ns = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="tgirl-smoke-"))
    print(f"workdir: {workdir}", file=sys.stderr)

    if ns.tools is None:
        stub = workdir / "stub_tools.py"
        stub.write_text(STUB_TOOLS_SOURCE)
        ns.tools = str(stub)

    probe_path = (
        Path(ns.probe_path) if ns.probe_path else workdir / "probe.npy"
    )
    base_url = f"http://127.0.0.1:{ns.port}"

    report: dict[str, Any] = {
        "workdir": str(workdir),
        "probe_path": str(probe_path),
        "model": ns.model,
        "turns_per_phase": ns.turns,
        "prompt": ns.prompt,
        "autosave_interval_s": ns.autosave_interval,
    }

    failure: Exception | None = None
    try:
        # Phase 1 — cold start, N turns, SIGTERM.
        report["phase1"] = _run_phase(
            phase="phase1",
            server_args=_build_cli_args(ns, probe_path),
            log_file=workdir / "server1.log",
            base_url=base_url,
            turns=ns.turns,
            prompt=ns.prompt,
            model=ns.model,
            expect_cached_at_startup=False,
        )

        if not probe_path.exists():
            raise AssertionError(
                f"probe file missing after shutdown: {probe_path}"
            )
        report["probe_file_size"] = probe_path.stat().st_size

        # Phase 2 — restart with --probe-load, assert continuity.
        phase2_args = _build_cli_args(ns, probe_path) + [
            "--probe-load",
            str(probe_path),
        ]
        report["phase2"] = _run_phase(
            phase="phase2",
            server_args=phase2_args,
            log_file=workdir / "server2.log",
            base_url=base_url,
            turns=ns.turns,
            prompt=ns.prompt,
            model=ns.model,
            expect_cached_at_startup=True,
        )

        report["continuity"] = _continuity(report["phase1"], report["phase2"])
        report["verdict"] = "PASS"
    except Exception as e:
        failure = e
        report["verdict"] = "FAIL"
        report["error"] = f"{type(e).__name__}: {e}"

    out = json.dumps(report, indent=2, default=str)
    if ns.output:
        Path(ns.output).write_text(out)
        print(f"report written: {ns.output}", file=sys.stderr)
    else:
        print(out)

    if failure is not None:
        print(f"SMOKE TEST FAILED: {failure}", file=sys.stderr)
    elif not ns.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0 if failure is None else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
