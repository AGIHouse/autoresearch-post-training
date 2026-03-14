#!/usr/bin/env python3
"""Print a comparison table of baseline vs trained eval results."""
import json
import os
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"

benchmarks = [
    ("MBPP", f"{results_dir}/eval_baseline.json", f"{results_dir}/eval_trained.json", "mbpp"),
    ("RLVR", f"{results_dir}/eval_baseline_rlvr.json", f"{results_dir}/eval_trained_rlvr.json", "rlvr"),
]

print("=" * 55)
for name, bf, tf, key in benchmarks:
    if not (os.path.exists(bf) and os.path.exists(tf)):
        continue
    base = json.load(open(bf))[key]
    trained = json.load(open(tf))[key]
    b, t = base["pass_at_1"], trained["pass_at_1"]
    print(f"{name:>6} Baseline: {b:.1%} ({base['passed']}/{base['total']})")
    print(f"{name:>6} Trained:  {t:.1%} ({trained['passed']}/{trained['total']})")
    print(f"{name:>6} Delta:    {(t-b)*100:+.1f} pp")
    print("-" * 55)
print("=" * 55)
