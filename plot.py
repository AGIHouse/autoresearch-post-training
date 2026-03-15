"""
Live-updating plot of pass@1 over iterations.
Polls results.tsv from the remote GPU and renders locally.

Usage:
    uv run plot.py              # polls every 30s
    uv run plot.py --interval 10  # polls every 10s
"""

import argparse
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # interactive backend

GPU_HOST = "ubuntu@38.128.232.129"
REMOTE_RESULTS = "autoresearch-post-training/results.tsv"


def fetch_results():
    """Read results.tsv from remote."""
    r = subprocess.run(
        f"ssh {GPU_HOST} cat {REMOTE_RESULTS}",
        shell=True, capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return None
    return r.stdout


def parse_results(tsv_text):
    """Parse results.tsv into lists."""
    iterations = []
    pass_at_1s = []
    statuses = []
    descriptions = []

    for line in tsv_text.strip().split("\n")[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            iterations.append(int(parts[0]))
            pass_at_1s.append(float(parts[1]))
            statuses.append(parts[3])
            descriptions.append(parts[4] if len(parts) > 4 else "")
        except (ValueError, IndexError):
            continue

    return iterations, pass_at_1s, statuses, descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title("autoresearch-post-training")

    print(f"Polling every {args.interval}s... (close window to stop)")

    while True:
        try:
            tsv = fetch_results()
            if tsv is None:
                time.sleep(args.interval)
                continue

            iterations, pass_at_1s, statuses, descriptions = parse_results(tsv)
            if not iterations:
                time.sleep(args.interval)
                continue

            ax.clear()

            # Plot all points
            keeps_x = [i for i, s in zip(iterations, statuses) if s == "keep"]
            keeps_y = [p for p, s in zip(pass_at_1s, statuses) if s == "keep"]
            discards_x = [i for i, s in zip(iterations, statuses) if s == "discard"]
            discards_y = [p for p, s in zip(pass_at_1s, statuses) if s == "discard"]
            crashes_x = [i for i, s in zip(iterations, statuses) if s == "crash"]
            crashes_y = [p for p, s in zip(pass_at_1s, statuses) if s == "crash"]

            ax.scatter(keeps_x, keeps_y, c="green", s=60, zorder=5, label="keep")
            ax.scatter(discards_x, discards_y, c="red", s=30, alpha=0.5, zorder=4, label="discard")
            ax.scatter(crashes_x, crashes_y, c="gray", s=20, alpha=0.3, zorder=3, marker="x", label="crash")

            # Best-so-far line
            best_so_far = []
            best = 0.0
            for p in pass_at_1s:
                if p > best:
                    best = p
                best_so_far.append(best)
            ax.plot(iterations, best_so_far, c="green", linewidth=2, alpha=0.7, label="best so far")

            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("pass@1", fontsize=12)
            ax.set_title(f"autoresearch-post-training  |  best={best:.4f}  |  {len(iterations)} experiments", fontsize=13)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
