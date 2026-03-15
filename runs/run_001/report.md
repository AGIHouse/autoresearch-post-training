# Run Report

**Date:** 2026-03-14 23:30:28  
**Model:** Qwen3.5-0.8B-Base + LoRA (r=16)  
**Dataset:** evalplus/mbppplus (300 train / 78 eval)  
**Config:** SFT 9steps + GRPO 23steps (lr=5e-06, G=8, beta=0)  
**Total time:** 25.9 min  

## Results

| Metric | Value |
|--------|-------|
| Baseline pass@1 | 30.0% |
| Best pass@1 | 30.0% (iter 0) |
| Δ vs baseline | +0.0% |
| Iterations run | 9 |
| Total SFT steps | 87 |
| Total GRPO steps | 192 |
| Total time | 25.9 min |

## Progress

![progress](progress.png)

## Iteration Log

| Iter | pass@1 | Δ baseline | SFT steps | GRPO steps | New best? |
|------|--------|-----------|-----------|------------|-----------|
| 0 | 30.0% | +0.0% | 0 | 0 | ★ |
| 1 | 20.0% | -10.0% | 9 | 23 |  |
| 2 | 20.0% | -10.0% | 10 | 17 |  |
| 3 | 20.0% | -10.0% | 10 | 18 |  |
| 4 | 16.7% | -13.3% | 10 | 21 |  |
| 5 | 13.3% | -16.7% | 9 | 22 |  |
| 6 | 13.3% | -16.7% | 10 | 24 |  |
| 7 | 16.7% | -13.3% | 9 | 23 |  |
| 8 | 20.0% | -10.0% | 10 | 21 |  |
| 9 | 26.7% | -3.3% | 10 | 23 |  |

## Analysis

Training did not improve over baseline (30.0%). Performance degraded to a minimum of 13.3% before partially recovering.

**Likely cause:** GRPO reward variance was zero throughout (all completions fail every test case), so group-relative advantages are undefined and no meaningful gradient flows. SFT at high LR overwrites learned behaviour faster than GRPO can recover it.

**Fixes to try:** increase GRPO budget (≥300s), reduce SFT LR (5e-5), or skip SFT (`--rl_only`) after the first iteration.
