#!/usr/bin/env python3
"""Analyze benchmark log files and extract step timing metrics."""
import re
import sys

logfile = sys.argv[1] if len(sys.argv) > 1 else "/home/vaishak/bench_1_baseline.log"
name = sys.argv[2] if len(sys.argv) > 2 else "experiment"

with open(logfile) as f:
    text = f.read()

# Extract step_time and step values from logged dicts
step_times = [float(m) for m in re.findall(r"'step_time': ([0-9.]+)", text)]
steps = [int(m) for m in re.findall(r"'step': (\d+)", text)]

if not step_times:
    print(f"No step_time data found in {logfile}")
    sys.exit(1)

# Skip first 3 steps (JIT compilation, warmup)
warmup_skip = 3
warm = step_times[warmup_skip:] if len(step_times) > warmup_skip else step_times

max_step = max(steps) if steps else len(step_times)
avg = sum(warm) / len(warm)
median = sorted(warm)[len(warm) // 2]

# Compute total tokens from last logged entry
tokens = [int(float(m)) for m in re.findall(r"'num_tokens': ([0-9.]+)", text)]
total_tokens = max(tokens) if tokens else 0

print(f"=== {name} ===")
print(f"Steps completed: {max_step}")
print(f"Step times logged: {len(step_times)} (skipping first {warmup_skip} for avg)")
print(f"Avg sec/step: {avg:.2f}")
print(f"Median sec/step: {median:.2f}")
print(f"Min sec/step: {min(warm):.2f}")
print(f"Max sec/step: {max(warm):.2f}")
print(f"Total tokens: {total_tokens:,}")
if total_tokens and max_step:
    print(f"Tokens/step: {total_tokens / max_step:.0f}")
print(f"Steps/10min (projected): {600 / avg:.0f}")

# Percentile analysis
p90 = sorted(warm)[int(len(warm) * 0.9)]
p95 = sorted(warm)[int(len(warm) * 0.95)]
p99 = sorted(warm)[int(len(warm) * 0.99)]
print(f"P50/P90/P95/P99: {median:.2f} / {p90:.2f} / {p95:.2f} / {p99:.2f}")

# Count outliers (>2x median)
outliers = [t for t in warm if t > median * 2]
print(f"Outliers (>2x median): {len(outliers)} / {len(warm)} ({100*len(outliers)/len(warm):.1f}%)")
if outliers:
    print(f"Avg without outliers: {sum(t for t in warm if t <= median * 2) / (len(warm) - len(outliers)):.2f}")
