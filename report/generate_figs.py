"""
Generate 10 publication-quality figures for the autoresearch-post-training report.
All figures are saved as PDFs for LaTeX inclusion (vector graphics).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

BASE = Path(__file__).parent.parent
r1 = load(BASE / "runs/run_001/experiment_log.jsonl")
r2 = load(BASE / "runs/run_002/experiment_log.jsonl")
r3 = load(BASE / "runs/run_003/experiment_log.jsonl")

# Style
COLORS = {
    "run1": "#2563EB",   # blue
    "run2": "#DC2626",   # red
    "run3": "#16A34A",   # green
    "baseline": "#6B7280",  # gray
    "pos": "#15803D",
    "neg": "#B91C1C",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ── Fig 1: Pass@1 vs Iteration — all runs ────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

for run, label, color, marker in [
    (r1, "Run 1 (SFT lr=5e-6, 9–10 SFT steps)", COLORS["run1"], "o"),
    (r2, "Run 2 (SFT lr=5e-5, higher LR)",       COLORS["run2"], "s"),
    (r3, "Run 3 (GRPO-only, no SFT)",             COLORS["run3"], "^"),
]:
    iters = [d["iteration"] for d in run]
    scores = [d["pass_at_1"] for d in run]
    ax.plot(iters, scores, color=color, marker=marker, linewidth=2,
            markersize=7, label=label)

ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
           label="Baseline (30.0%)")
ax.set_xlabel("Iteration")
ax.set_ylabel("pass@1 (%)")
ax.set_title("Pass@1 Over Training Iterations — All Runs")
ax.set_ylim(0, 42)
ax.set_xticks(range(10))
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "fig1_pass_all_runs.pdf")
plt.close(fig)
print("Fig 1 done")

# ── Fig 2: Degradation depth + recovery (run 1 annotated) ────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

iters = [d["iteration"] for d in r1]
scores = [d["pass_at_1"] for d in r1]
ax.fill_between(iters, 30.0, scores,
                where=[s < 30.0 for s in scores],
                color="#FCA5A5", alpha=0.5, label="Below baseline")
ax.fill_between(iters, 30.0, scores,
                where=[s >= 30.0 for s in scores],
                color="#86EFAC", alpha=0.5, label="At/above baseline")
ax.plot(iters, scores, color=COLORS["run1"], marker="o", linewidth=2, markersize=7, zorder=3)
ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Baseline 30.0%")

# Annotate minimum
min_idx = int(np.argmin(scores))
ax.annotate(f"Min: {scores[min_idx]:.1f}%",
            xy=(iters[min_idx], scores[min_idx]),
            xytext=(iters[min_idx]+0.5, scores[min_idx]-3),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10)
ax.annotate(f"Best recovered: {scores[-1]:.1f}%",
            xy=(iters[-1], scores[-1]),
            xytext=(iters[-1]-2.5, scores[-1]+4),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10)

ax.set_xlabel("Iteration")
ax.set_ylabel("pass@1 (%)")
ax.set_title("Run 1: Degradation and Partial Recovery")
ax.set_ylim(0, 42)
ax.set_xticks(range(10))
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "fig2_degradation_run1.pdf")
plt.close(fig)
print("Fig 2 done")

# ── Fig 3: GRPO steps per iteration — grouped bar ────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4.5))

max_iter = 9
x = np.arange(1, max_iter + 1)  # iterations 1..9

def grpo_arr(run, n):
    vals = []
    for d in run[1:]:  # skip baseline
        vals.append(d["grpo_steps_run"])
    # pad to n
    return vals + [0] * (n - len(vals))

g1 = grpo_arr(r1, max_iter)
g2 = grpo_arr(r2, max_iter)
g3 = grpo_arr(r3, max_iter)

w = 0.28
ax.bar(x - w, g1, w, color=COLORS["run1"], label="Run 1", alpha=0.85)
ax.bar(x,     g2, w, color=COLORS["run2"], label="Run 2", alpha=0.85)
ax.bar(x + w, g3, w, color=COLORS["run3"], label="Run 3", alpha=0.85)

ax.set_xlabel("Iteration")
ax.set_ylabel("GRPO Steps")
ax.set_title("GRPO Gradient Steps per Iteration Across Runs")
ax.set_xticks(x)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig3_grpo_steps_bar.pdf")
plt.close(fig)
print("Fig 3 done")

# ── Fig 4: Cumulative total steps vs pass@1 ──────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

for run, label, color, marker in [
    (r1, "Run 1", COLORS["run1"], "o"),
    (r2, "Run 2", COLORS["run2"], "s"),
    (r3, "Run 3", COLORS["run3"], "^"),
]:
    cum = 0
    cum_steps = []
    for d in run:
        cum += d["sft_steps_run"] + d["grpo_steps_run"]
        cum_steps.append(cum)
    scores = [d["pass_at_1"] for d in run]
    ax.plot(cum_steps, scores, color=color, marker=marker, linewidth=2,
            markersize=7, label=label)
    for cs, sc in zip(cum_steps, scores):
        ax.annotate(f"{sc:.0f}%", (cs, sc), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7.5, color=color)

ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
           label="Baseline 30%")
ax.set_xlabel("Cumulative Gradient Steps (SFT + GRPO)")
ax.set_ylabel("pass@1 (%)")
ax.set_title("Performance vs. Total Compute (Gradient Steps)")
ax.set_ylim(0, 42)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig4_cumsteps_vs_pass.pdf")
plt.close(fig)
print("Fig 4 done")

# ── Fig 5: Wall-clock time vs pass@1 ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

for run, label, color, marker in [
    (r1, "Run 1", COLORS["run1"], "o"),
    (r2, "Run 2", COLORS["run2"], "s"),
    (r3, "Run 3", COLORS["run3"], "^"),
]:
    times = [d["elapsed_s"] / 60 for d in run]  # minutes
    scores = [d["pass_at_1"] for d in run]
    ax.plot(times, scores, color=color, marker=marker, linewidth=2,
            markersize=7, label=label)

ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
           label="Baseline 30%")
ax.set_xlabel("Elapsed Wall-Clock Time (minutes)")
ax.set_ylabel("pass@1 (%)")
ax.set_title("Performance vs. Wall-Clock Time")
ax.set_ylim(0, 42)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig5_wallclock_vs_pass.pdf")
plt.close(fig)
print("Fig 5 done")

# ── Fig 6: Delta heatmap ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 3.5))

# Build matrix: rows = runs, cols = iterations 0..9
runs_data = [r1, r2, r3]
run_labels = ["Run 1", "Run 2", "Run 3"]
n_iters = 10
mat = np.full((3, n_iters), np.nan)
for ri, run in enumerate(runs_data):
    for d in run:
        it = d["iteration"]
        mat[ri, it] = d["delta"]

im = ax.imshow(mat, cmap="RdYlGn", vmin=-20, vmax=10, aspect="auto")
ax.set_xticks(range(n_iters))
ax.set_xticklabels([str(i) for i in range(n_iters)])
ax.set_yticks(range(3))
ax.set_yticklabels(run_labels)
ax.set_xlabel("Iteration")
ax.set_title("Delta from Baseline (percentage points) — Heatmap")

for ri in range(3):
    for ci in range(n_iters):
        val = mat[ri, ci]
        if not np.isnan(val):
            ax.text(ci, ri, f"{val:+.1f}", ha="center", va="center",
                    fontsize=9, color="black",
                    fontweight="bold" if val >= 0 else "normal")
        else:
            ax.text(ci, ri, "—", ha="center", va="center",
                    fontsize=9, color="#9CA3AF")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Δ pass@1 (pp)")
fig.tight_layout()
fig.savefig(OUT / "fig6_delta_heatmap.pdf")
plt.close(fig)
print("Fig 6 done")

# ── Fig 7: Per-iteration time cost ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

for run, label, color, marker in [
    (r1, "Run 1", COLORS["run1"], "o"),
    (r2, "Run 2", COLORS["run2"], "s"),
    (r3, "Run 3", COLORS["run3"], "^"),
]:
    elapsed = [d["elapsed_s"] for d in run]
    # time per iteration = difference
    iter_times = [elapsed[0]] + [elapsed[i] - elapsed[i-1] for i in range(1, len(elapsed))]
    iters = [d["iteration"] for d in run]
    ax.plot(iters, [t/60 for t in iter_times], color=color, marker=marker,
            linewidth=2, markersize=7, label=label)

ax.set_xlabel("Iteration")
ax.set_ylabel("Time per Iteration (minutes)")
ax.set_title("Wall-Clock Time per Iteration")
ax.set_xticks(range(10))
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig7_iter_time.pdf")
plt.close(fig)
print("Fig 7 done")

# ── Fig 8: SFT vs GRPO steps scatter coloured by pass@1 ──────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

all_sft, all_grpo, all_pass, all_run = [], [], [], []
run_map = [(r1, "Run 1", "o"), (r2, "Run 2", "s"), (r3, "Run 3", "^")]
for run, rname, mrk in run_map:
    for d in run[1:]:  # skip baseline
        all_sft.append(d["sft_steps_run"])
        all_grpo.append(d["grpo_steps_run"])
        all_pass.append(d["pass_at_1"])
        all_run.append((rname, mrk))

sc = ax.scatter(all_sft, all_grpo, c=all_pass, cmap="RdYlGn",
                vmin=10, vmax=32, s=120, zorder=3, edgecolors="k", linewidths=0.5)
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("pass@1 (%)")
ax.set_xlabel("SFT Steps in Iteration")
ax.set_ylabel("GRPO Steps in Iteration")
ax.set_title("SFT vs. GRPO Steps (colour = pass@1)")

# Legend for runs
for run, rname, mrk in run_map:
    ax.scatter([], [], marker=mrk, color="gray", label=rname, s=80)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig8_sft_grpo_scatter.pdf")
plt.close(fig)
print("Fig 8 done")

# ── Fig 9: Run summary bar chart ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))

summary = [
    dict(run="Run 1", iters=9, total_sft=87, total_grpo=192,
         best=30.0, worst=13.3, final=26.7, time=25.9),
    dict(run="Run 2", iters=4, total_sft=39, total_grpo=211,
         best=30.0, worst=13.3, final=30.0, time=29.7),
    dict(run="Run 3", iters=2, total_sft=0,  total_grpo=77,
         best=30.0, worst=20.0, final=20.0, time=30.6),
]
run_names = [s["run"] for s in summary]
run_colors = [COLORS["run1"], COLORS["run2"], COLORS["run3"]]
x = np.arange(3)

# Panel A: total steps
ax = axes[0]
ax.bar(x - 0.2, [s["total_sft"] for s in summary],  0.4, color="#60A5FA", label="SFT")
ax.bar(x + 0.2, [s["total_grpo"] for s in summary], 0.4, color="#F59E0B", label="GRPO")
ax.set_xticks(x); ax.set_xticklabels(run_names)
ax.set_ylabel("Total Steps"); ax.set_title("A  Total Gradient Steps")
ax.legend()

# Panel B: pass@1 range (worst – best – final)
ax = axes[1]
for xi, s in enumerate(summary):
    ax.plot([xi, xi], [s["worst"], s["best"]], color=run_colors[xi], linewidth=4, alpha=0.4)
    ax.scatter(xi, s["worst"],  color=run_colors[xi], marker="v", s=100, zorder=3)
    ax.scatter(xi, s["best"],   color=run_colors[xi], marker="^", s=100, zorder=3)
    ax.scatter(xi, s["final"],  color=run_colors[xi], marker="D", s=100, zorder=3, label=f"{s['run']} final={s['final']:.0f}%")
ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(run_names)
ax.set_ylabel("pass@1 (%)"); ax.set_title("B  Score Range (▼worst, ▲best, ◆final)")
ax.set_ylim(0, 42)
ax.legend(fontsize=8)

# Panel C: wall-clock time
ax = axes[2]
ax.bar(x, [s["time"] for s in summary], color=run_colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(run_names)
ax.set_ylabel("Total Time (min)"); ax.set_title("C  Total Wall-Clock Time")
for xi, s in enumerate(summary):
    ax.text(xi, s["time"] + 0.5, f"{s['time']:.1f}m", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig9_run_summary.pdf")
plt.close(fig)
print("Fig 9 done")

# ── Fig 10: Pass@1 distribution / variance analysis ──────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: box-like plot showing score spread per run (post-baseline iters)
ax = axes[0]
post_baseline = [
    [d["pass_at_1"] for d in r1[1:]],
    [d["pass_at_1"] for d in r2[1:]],
    [d["pass_at_1"] for d in r3[1:]],
]
bp = ax.boxplot(post_baseline, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], run_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Baseline 30%")
ax.set_xticklabels(["Run 1\n(9 iters)", "Run 2\n(4 iters)", "Run 3\n(2 iters)"])
ax.set_ylabel("pass@1 (%)")
ax.set_title("A  Score Distribution (Post-Baseline Iterations)")
ax.set_ylim(0, 42)
ax.legend()

# Right: running mean of pass@1 across all iterations combined
ax = axes[1]
# Combine all runs chronologically and show rolling trend
all_scores_r1 = [d["pass_at_1"] for d in r1]
window = 3
rolling = [np.mean(all_scores_r1[max(0,i-window+1):i+1]) for i in range(len(all_scores_r1))]
ax.plot(range(len(all_scores_r1)), all_scores_r1, color=COLORS["run1"],
        marker="o", linewidth=1.5, alpha=0.5, label="Raw pass@1")
ax.plot(range(len(rolling)), rolling, color=COLORS["run1"],
        linewidth=2.5, label=f"Rolling mean (w={window})")
ax.axhline(30.0, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Baseline")
ax.set_xlabel("Iteration")
ax.set_ylabel("pass@1 (%)")
ax.set_title("B  Run 1: Raw vs. Rolling Mean pass@1")
ax.set_ylim(0, 42)
ax.set_xticks(range(10))
ax.legend()

fig.tight_layout()
fig.savefig(OUT / "fig10_distribution.pdf")
plt.close(fig)
print("Fig 10 done")

print(f"\nAll figures saved to {OUT}/")
