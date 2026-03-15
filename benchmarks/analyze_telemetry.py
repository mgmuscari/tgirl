#!/usr/bin/env python
"""Analyze tgirl BFCL telemetry JSONL for hyperparameter tuning insights.

Reads telemetry produced by run_bfcl.py and runs six analyses:
  A. Grammar freedom vs accuracy
  B. Temperature curve effectiveness
  C. OT engagement vs accuracy
  D. Token confidence at decision points
  E. Freeform thinking length vs accuracy
  F. Boolean fix impact

Usage:
    python benchmarks/analyze_telemetry.py result/tgirl-9b/BFCL_v4_simple_python_telemetry.jsonl
    python benchmarks/analyze_telemetry.py --plot result/tgirl-9b/BFCL_v4_simple_python_telemetry.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def partition(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split entries into PASS and FAIL groups."""
    passes = [e for e in entries if e.get("ast_result") == "PASS"]
    fails = [e for e in entries if e.get("ast_result", "").startswith("FAIL")]
    return passes, fails


def safe_mean(vals: list[float | int]) -> float:
    return statistics.mean(vals) if vals else 0.0


def safe_median(vals: list[float | int]) -> float:
    return statistics.median(vals) if vals else 0.0


def analysis_a_grammar_freedom(entries: list[dict]) -> dict[str, Any]:
    """A. Grammar freedom vs accuracy."""
    passes, fails = partition(entries)

    def freedom_stats(group: list[dict]) -> dict[str, float]:
        all_counts: list[float] = []
        for e in group:
            counts = e.get("grammar_valid_counts", [])
            all_counts.extend(counts)
        return {
            "mean": round(safe_mean(all_counts), 2),
            "median": round(safe_median(all_counts), 2),
            "count": len(group),
        }

    result = {
        "pass": freedom_stats(passes),
        "fail": freedom_stats(fails),
    }

    print("\n--- A. Grammar Freedom vs Accuracy ---")
    print(f"  PASS ({result['pass']['count']} entries): "
          f"mean={result['pass']['mean']}, median={result['pass']['median']}")
    print(f"  FAIL ({result['fail']['count']} entries): "
          f"mean={result['fail']['mean']}, median={result['fail']['median']}")
    return result


def analysis_b_temperature(entries: list[dict]) -> dict[str, Any]:
    """B. Temperature curve effectiveness."""
    passes, fails = partition(entries)

    # Bin by grammar freedom decile, compute pass rate
    all_with_labels: list[tuple[int, bool]] = []
    for e in entries:
        is_pass = e.get("ast_result") == "PASS"
        counts = e.get("grammar_valid_counts", [])
        temps = e.get("temperatures_applied", [])
        for c, _t in zip(counts, temps):
            all_with_labels.append((c, is_pass))

    if not all_with_labels:
        print("\n--- B. Temperature Curve ---")
        print("  No data available")
        return {}

    sorted_data = sorted(all_with_labels, key=lambda x: x[0])
    n = len(sorted_data)
    decile_size = max(1, n // 10)

    decile_results = []
    for d in range(10):
        start = d * decile_size
        end = start + decile_size if d < 9 else n
        chunk = sorted_data[start:end]
        if not chunk:
            continue
        freedom_range = (chunk[0][0], chunk[-1][0])
        pass_rate = sum(1 for _, p in chunk if p) / len(chunk) * 100
        decile_results.append({
            "decile": d + 1,
            "freedom_range": freedom_range,
            "pass_rate": round(pass_rate, 1),
            "count": len(chunk),
        })

    # Temperature stats by group
    def temp_stats(group: list[dict]) -> dict[str, float]:
        all_temps: list[float] = []
        for e in group:
            temps = e.get("temperatures_applied", [])
            all_temps.extend(t for t in temps if t >= 0)
        return {
            "mean": round(safe_mean(all_temps), 4),
            "median": round(safe_median(all_temps), 4),
        }

    result = {
        "deciles": decile_results,
        "pass_temps": temp_stats(passes),
        "fail_temps": temp_stats(fails),
    }

    print("\n--- B. Temperature Curve Effectiveness ---")
    print(f"  PASS temps: mean={result['pass_temps']['mean']}, "
          f"median={result['pass_temps']['median']}")
    print(f"  FAIL temps: mean={result['fail_temps']['mean']}, "
          f"median={result['fail_temps']['median']}")
    print("  Pass rate by freedom decile:")
    for d in decile_results:
        print(f"    D{d['decile']}: freedom {d['freedom_range']}, "
              f"pass_rate={d['pass_rate']}%, n={d['count']}")
    return result


def analysis_c_ot_engagement(entries: list[dict]) -> dict[str, Any]:
    """C. OT engagement vs accuracy."""
    passes, fails = partition(entries)

    def ot_stats(group: list[dict]) -> dict[str, Any]:
        total_tokens = 0
        ot_engaged = 0
        all_iterations: list[int] = []
        all_distances: list[float] = []
        for e in group:
            reasons = e.get("ot_bypass_reasons", [])
            iters = e.get("ot_iterations", [])
            dists = e.get("wasserstein_distances", [])
            for reason, it, dist in zip(reasons, iters, dists):
                total_tokens += 1
                if reason is None:
                    ot_engaged += 1
                    all_iterations.append(it)
                    all_distances.append(dist)
        engaged_ratio = ot_engaged / total_tokens if total_tokens else 0
        return {
            "total_tokens": total_tokens,
            "ot_engaged": ot_engaged,
            "engaged_ratio": round(engaged_ratio, 3),
            "mean_iterations": round(safe_mean(all_iterations), 2),
            "mean_wasserstein": round(safe_mean(all_distances), 4),
            "count": len(group),
        }

    result = {
        "pass": ot_stats(passes),
        "fail": ot_stats(fails),
    }

    # Bypass reason breakdown
    reason_counts: dict[str, int] = {}
    for e in entries:
        for reason in e.get("ot_bypass_reasons", []):
            key = reason if reason else "engaged"
            reason_counts[key] = reason_counts.get(key, 0) + 1

    result["bypass_reasons"] = reason_counts

    print("\n--- C. OT Engagement vs Accuracy ---")
    for label in ("pass", "fail"):
        s = result[label]
        print(f"  {label.upper()} ({s['count']} entries): "
              f"engaged={s['engaged_ratio']:.1%}, "
              f"mean_iters={s['mean_iterations']}, "
              f"mean_W={s['mean_wasserstein']}")
    print("  Bypass reasons:")
    for reason, count in sorted(reason_counts.items()):
        print(f"    {reason}: {count}")
    return result


def analysis_d_confidence(entries: list[dict]) -> dict[str, Any]:
    """D. Token confidence at decision points."""
    passes, fails = partition(entries)
    freedom_threshold = 10  # positions with real choices

    def confidence_stats(group: list[dict]) -> dict[str, float]:
        all_lps: list[float] = []
        for e in group:
            counts = e.get("grammar_valid_counts", [])
            lps = e.get("token_log_probs", [])
            for c, lp in zip(counts, lps):
                if c > freedom_threshold and lp > float("-inf"):
                    all_lps.append(lp)
        return {
            "mean": round(safe_mean(all_lps), 4),
            "min": round(min(all_lps), 4) if all_lps else 0.0,
            "count": len(all_lps),
        }

    result = {
        "pass": confidence_stats(passes),
        "fail": confidence_stats(fails),
        "freedom_threshold": freedom_threshold,
    }

    print("\n--- D. Token Confidence at Decision Points ---")
    print(f"  (freedom > {freedom_threshold} tokens)")
    for label in ("pass", "fail"):
        s = result[label]
        print(f"  {label.upper()}: mean_logprob={s['mean']}, "
              f"min_logprob={s['min']}, n_tokens={s['count']}")
    return result


def analysis_e_freeform_length(entries: list[dict]) -> dict[str, Any]:
    """E. Freeform thinking length vs accuracy."""
    passes, fails = partition(entries)

    def length_stats(group: list[dict]) -> dict[str, float]:
        lengths = [e.get("freeform_tokens_before", 0) for e in group]
        return {
            "mean": round(safe_mean(lengths), 1),
            "median": round(safe_median(lengths), 1),
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "count": len(group),
        }

    result = {
        "pass": length_stats(passes),
        "fail": length_stats(fails),
    }

    print("\n--- E. Freeform Thinking Length vs Accuracy ---")
    for label in ("pass", "fail"):
        s = result[label]
        print(f"  {label.upper()} ({s['count']}): "
              f"mean={s['mean']}, median={s['median']}, "
              f"range=[{s['min']}, {s['max']}]")
    return result


def analysis_f_boolean_impact(entries: list[dict]) -> dict[str, Any]:
    """F. Boolean fix impact."""
    bool_failures = []
    for e in entries:
        ast_result = e.get("ast_result", "")
        if "type" in ast_result.lower() and ("bool" in ast_result.lower()
                                              or "true" in ast_result.lower()
                                              or "false" in ast_result.lower()):
            bool_failures.append(e["id"])

    # Also check for literal "true"/"false" in hy_source (pre-fix indicator)
    lowercase_bool_entries = []
    for e in entries:
        hy = e.get("hy_source", "")
        if " true" in hy or " false" in hy or "(true" in hy or "(false" in hy:
            lowercase_bool_entries.append(e["id"])

    total = len(entries)
    _, fails = partition(entries)

    result = {
        "bool_type_failures": len(bool_failures),
        "lowercase_bool_in_hy": len(lowercase_bool_entries),
        "total_entries": total,
        "total_fails": len(fails),
        "estimated_accuracy_gain_pct": round(
            len(bool_failures) / total * 100, 1
        ) if total else 0,
    }

    print("\n--- F. Boolean Fix Impact ---")
    print(f"  Boolean-related type failures: {result['bool_type_failures']}")
    print(f"  Entries with lowercase bool in Hy: {result['lowercase_bool_in_hy']}")
    print(f"  Total fails: {result['total_fails']} / {result['total_entries']}")
    print(f"  Estimated accuracy gain from fix: +{result['estimated_accuracy_gain_pct']}%")
    return result


def make_plots(entries: list[dict], output_dir: Path) -> None:
    """Generate matplotlib plots (optional)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib not installed — skipping plots")
        return

    passes, fails = partition(entries)

    # Plot 1: Grammar freedom distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pass_counts = []
    for e in passes:
        pass_counts.extend(e.get("grammar_valid_counts", []))
    fail_counts = []
    for e in fails:
        fail_counts.extend(e.get("grammar_valid_counts", []))

    if pass_counts:
        axes[0].hist(pass_counts, bins=50, alpha=0.7, label="PASS", color="green")
    if fail_counts:
        axes[0].hist(fail_counts, bins=50, alpha=0.7, label="FAIL", color="red")
    axes[0].set_xlabel("Grammar Valid Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Grammar Freedom Distribution")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Plot 2: Freeform length distribution
    pass_lengths = [e.get("freeform_tokens_before", 0) for e in passes]
    fail_lengths = [e.get("freeform_tokens_before", 0) for e in fails]
    if pass_lengths:
        axes[1].hist(pass_lengths, bins=30, alpha=0.7, label="PASS", color="green")
    if fail_lengths:
        axes[1].hist(fail_lengths, bins=30, alpha=0.7, label="FAIL", color="red")
    axes[1].set_xlabel("Freeform Tokens Before Tool Call")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Thinking Length Distribution")
    axes[1].legend()

    plot_path = output_dir / "telemetry_analysis.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[+] Plots saved to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tgirl BFCL telemetry")
    parser.add_argument("telemetry_file", type=Path, help="Path to telemetry JSONL")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for summary.json and plots")
    args = parser.parse_args()

    if not args.telemetry_file.exists():
        print(f"Error: {args.telemetry_file} not found", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(args.telemetry_file)
    if not entries:
        print("Error: no entries in telemetry file", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.telemetry_file.parent

    passes, fails = partition(entries)
    total = len(entries)
    no_tool = sum(1 for e in entries if e.get("ast_result") == "NO_TOOL_CALL")
    errors = sum(1 for e in entries if e.get("ast_result", "").startswith("ERROR"))

    print("=" * 60)
    print(f"  tgirl Telemetry Analysis — {args.telemetry_file.name}")
    print("=" * 60)
    print(f"  Total entries:   {total}")
    print(f"  PASS:            {len(passes)} ({round(100*len(passes)/total)}%)")
    print(f"  FAIL:            {len(fails)} ({round(100*len(fails)/total)}%)")
    print(f"  NO_TOOL_CALL:    {no_tool}")
    print(f"  ERROR:           {errors}")

    summary: dict[str, Any] = {
        "file": str(args.telemetry_file),
        "total": total,
        "pass": len(passes),
        "fail": len(fails),
        "no_tool_call": no_tool,
        "errors": errors,
    }

    summary["a_grammar_freedom"] = analysis_a_grammar_freedom(entries)
    summary["b_temperature"] = analysis_b_temperature(entries)
    summary["c_ot_engagement"] = analysis_c_ot_engagement(entries)
    summary["d_confidence"] = analysis_d_confidence(entries)
    summary["e_freeform_length"] = analysis_e_freeform_length(entries)
    summary["f_boolean_impact"] = analysis_f_boolean_impact(entries)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Summary written to {summary_path}")

    if args.plot:
        make_plots(entries, output_dir)


if __name__ == "__main__":
    main()
