"""
Plot autoresearch training progress (karpathy style) and write a markdown report.

Reads experiment_log.jsonl and produces:
  - progress.png   — the graph
  - report.md      — markdown report embedding the graph + results table

Usage:
    python plot.py                                 # uses ./experiment_log.jsonl
    python plot.py --run_dir runs/run_002
    python plot.py --log path/to/log.jsonl --out path/to/progress.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def load_log(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def make_graph(entries: list[dict], out_path: str):
    iterations = [e["iteration"] for e in entries]
    scores     = [e["pass_at_1"] for e in entries]
    baseline   = entries[0]["pass_at_1"]

    # Running maximum (the frontier)
    best = -1
    best_entries = []
    frontier = []
    for e in entries:
        if e["pass_at_1"] > best:
            best = e["pass_at_1"]
            best_entries.append(e)
        frontier.append(best)

    n_total = len(entries) - 1
    n_kept  = len(best_entries) - 1

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # All iterations — faint grey dots
    ax.scatter(
        iterations, scores,
        color="#cccccc", edgecolors="#aaaaaa", s=70, zorder=2,
        linewidths=0.7, label="all iterations",
    )

    # Step-function frontier
    step_x = [iterations[0]]
    step_y = [frontier[0]]
    for x, y in zip(iterations[1:], frontier[1:]):
        step_x.extend([x, x])
        step_y.extend([step_y[-1], y])
    ax.plot(
        step_x, step_y,
        color="#27ae60", linewidth=2.5, zorder=3,
        solid_capstyle="round", label="best so far",
    )

    # New-best dots — green
    best_x = [e["iteration"] for e in best_entries]
    best_y = [e["pass_at_1"] for e in best_entries]
    ax.scatter(
        best_x, best_y,
        color="#2ecc71", edgecolors="black", s=130, zorder=5,
        linewidths=1.0, label="new best",
    )

    # Annotations on new-best dots
    for e in best_entries:
        desc = e["description"]
        if len(desc) > 48:
            desc = desc[:45] + "…"
        ax.annotate(
            desc,
            xy=(e["iteration"], e["pass_at_1"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=7.5,
            color="#1a5c38",
            rotation=25,
            va="bottom",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#fafafa")],
        )

    # Baseline reference line
    ax.axhline(
        baseline, color="#e74c3c", linestyle="--", linewidth=1.2,
        alpha=0.75, zorder=1, label=f"baseline ({baseline:.1f}%)",
    )

    # Axes
    y_min = min(scores)
    y_max = max(scores)
    margin = max((y_max - y_min) * 0.3, 4.0)
    ax.set_ylim(max(0, y_min - margin), min(100, y_max + margin * 2.5))
    ax.set_xlim(-0.5, max(iterations) + 1)
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("pass@1  (%)", fontsize=13)
    ax.set_title(
        f"autoresearch progress — {n_total} iterations, {n_kept} improvements\n"
        f"Qwen3.5-0.8B-Base + SFT → GRPO on MBPP++",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linewidth=0.7)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.88)

    # Summary box
    best_score = max(scores)
    delta = best_score - baseline
    ax.text(
        0.02, 0.97,
        f"baseline  {baseline:.1f}%\nbest      {best_score:.1f}%\nΔ         {delta:+.1f}%",
        transform=ax.transAxes,
        fontsize=9, va="top", ha="left", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph  → {out_path}")


def make_report(entries: list[dict], png_path: str, report_path: str):
    baseline   = entries[0]["pass_at_1"]
    best_score = max(e["pass_at_1"] for e in entries)
    best_iter  = max(entries, key=lambda e: e["pass_at_1"])
    n_iters    = len(entries) - 1
    total_sft  = sum(e.get("sft_steps_run", 0) for e in entries)
    total_grpo = sum(e.get("grpo_steps_run", 0) for e in entries)
    elapsed    = entries[-1].get("elapsed_s", 0)
    timestamp  = entries[0].get("timestamp", "")[:19].replace("T", " ")

    # Extract config from first training entry if present
    cfg_entry = entries[1] if len(entries) > 1 else entries[0]
    desc0 = cfg_entry.get("description", "")

    png_rel = os.path.basename(png_path)

    lines = [
        f"# Run Report",
        f"",
        f"**Date:** {timestamp}  ",
        f"**Model:** Qwen3.5-0.8B-Base + LoRA (r=16)  ",
        f"**Dataset:** evalplus/mbppplus (300 train / 78 eval)  ",
        f"**Config:** {desc0}  ",
        f"**Total time:** {elapsed/60:.1f} min  ",
        f"",
        f"## Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Baseline pass@1 | {baseline:.1f}% |",
        f"| Best pass@1 | {best_score:.1f}% (iter {best_iter['iteration']}) |",
        f"| Δ vs baseline | {best_score - baseline:+.1f}% |",
        f"| Iterations run | {n_iters} |",
        f"| Total SFT steps | {total_sft} |",
        f"| Total GRPO steps | {total_grpo} |",
        f"| Total time | {elapsed/60:.1f} min |",
        f"",
        f"## Progress",
        f"",
        f"![progress]({png_rel})",
        f"",
        f"## Iteration Log",
        f"",
        f"| Iter | pass@1 | Δ baseline | SFT steps | GRPO steps | New best? |",
        f"|------|--------|-----------|-----------|------------|-----------|",
    ]

    for e in entries:
        star = "★" if e.get("is_best") else ""
        lines.append(
            f"| {e['iteration']} | {e['pass_at_1']:.1f}% | {e['delta']:+.1f}% "
            f"| {e.get('sft_steps_run', 0)} | {e.get('grpo_steps_run', 0)} | {star} |"
        )

    lines += [
        f"",
        f"## Analysis",
        f"",
    ]

    if best_score > baseline:
        lines.append(
            f"Training improved pass@1 from **{baseline:.1f}%** to **{best_score:.1f}%** "
            f"(+{best_score-baseline:.1f}pp) at iteration {best_iter['iteration']}."
        )
    else:
        worst = min(e["pass_at_1"] for e in entries)
        lines += [
            f"Training did not improve over baseline ({baseline:.1f}%). "
            f"Performance degraded to a minimum of {worst:.1f}% before partially recovering.",
            f"",
            f"**Likely cause:** GRPO reward variance was zero throughout (all completions "
            f"fail every test case), so group-relative advantages are undefined and no "
            f"meaningful gradient flows. SFT at high LR overwrites learned behaviour faster "
            f"than GRPO can recover it.",
            f"",
            f"**Fixes to try:** increase GRPO budget (≥300s), reduce SFT LR (5e-5), "
            f"or skip SFT (`--rl_only`) after the first iteration.",
        ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report → {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None,
                        help="run directory (sets --log, --out, --report automatically)")
    parser.add_argument("--log",    default="./experiment_log.jsonl")
    parser.add_argument("--out",    default="./progress.png")
    parser.add_argument("--report", default="./report.md")
    args = parser.parse_args()

    if args.run_dir:
        args.log    = os.path.join(args.run_dir, "experiment_log.jsonl")
        args.out    = os.path.join(args.run_dir, "progress.png")
        args.report = os.path.join(args.run_dir, "report.md")

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}")
        return

    entries = load_log(args.log)
    if len(entries) < 2:
        print(f"Only {len(entries)} entries — need at least 2 to plot.")
        return

    make_graph(entries, args.out)
    make_report(entries, args.out, args.report)

    # Print summary table to stdout
    print(f"\n{'Iter':>4}  {'pass@1':>7}  {'delta':>7}  {'best?':>5}  description")
    print("─" * 75)
    for e in entries:
        star = "★" if e.get("is_best") else " "
        print(f"{e['iteration']:>4}  {e['pass_at_1']:>6.1f}%  {e['delta']:>+6.1f}%"
              f"    {star}    {e['description'][:42]}")


if __name__ == "__main__":
    main()
