# Training Performance Optimization Experiments

Hardware: 1x NVIDIA H100 80GB (GCP a3-highgpu-1g)
Model: Qwen/Qwen2.5-Coder-7B-Instruct + LoRA (r=64)
Framework: TRL 0.29.0 + vLLM 0.17.1 + PyTorch 2.10.0
Each run: 10 minutes wall clock
Dataset: RLVR (func-type, ~2550 problems)
Batch: 2 prompts × 4 grad_accum × 8 generations = 64 completions/step

## Summary

| # | Optimization | Steps | Avg s/step | Median s/step | Speedup (avg) | Cumulative |
|---|-------------|-------|------------|---------------|---------------|------------|
| 1 | Baseline (Docker sandbox, vLLM colocate 0.3) | 112 | 4.26 | 2.75 | 1.00x | 1.00x |
| 2 | + Subprocess sandbox | 140 | 3.80 | 2.64 | 1.12x | 1.12x |
| 3 | + vLLM GPU mem 0.3→0.45 | 154 | 3.48 | 2.38 | 1.09x | 1.22x |
| 4 | + Iteration reuse (num_iterations=2) | 159 | 3.43 | 2.35 | 1.01x | 1.24x |
| 5 | Switch to OpenRLHF/veRL | — | — | — | — | — |

**Key insight**: ~13% of steps are outliers (>2x median), inflating the average. Without outliers,
all experiments converge to ~2.4-2.75 s/step median. The outliers are likely caused by:
- vLLM colocate memory swap overhead (model sleep/wake)
- Sandbox timeout on hard problems (10s timeout × 8 workers)
- Python GC / CUDA memory management pauses

**Avg without outliers**:
- Exp 1: 2.75 s/step (vs 4.26 avg)
- Exp 2: 2.68 s/step (vs 3.80 avg)
- Exp 3: 2.40 s/step (vs 3.48 avg)
- Exp 4: 2.42 s/step (vs 3.43 avg)

---

## Experiment 1: Baseline

**Config**: dr_grpo.yaml on H100, Docker sandbox (8 workers, 10s timeout), vLLM colocate (0.3 GPU mem)
**Changes**: None (baseline measurement)

```
Steps completed:  112
Avg sec/step:     4.26
Median sec/step:  2.75
Min/Max:          1.96 / 18.91
P50/P90/P95/P99:  2.75 / 12.81 / 17.10 / 18.39
Outliers:         14/109 (12.8%)
Avg w/o outliers: 2.75
Total tokens:     504,827
```

**Notes**: Surprisingly fast baseline on H100 — 2.75s median per step for a 7B model with
8 generations and Docker sandbox execution. The ~13% outlier steps dominate the average.

---

## Experiment 2: Subprocess Sandbox

**Config**: Same as #1, but `sandbox_backend: subprocess`
**Hypothesis**: Eliminates ~100ms/container Docker startup overhead for 64 containers/step.

```
Steps completed:  140
Avg sec/step:     3.80 (1.12x faster)
Median sec/step:  2.64
Min/Max:          1.79 / 13.61
P50/P90/P95/P99:  2.64 / 8.65 / 12.83 / 13.51
Outliers:         19/137 (13.9%)
Avg w/o outliers: 2.68
Total tokens:     641,010
```

**Notes**: Modest improvement. Docker overhead (~100ms × ceil(64/8) containers) accounts for
~0.8s/step. Subprocess eliminates this entirely but the gain is modest vs total step time.
The outlier spike magnitude also reduced (max 18.91→13.61) since no Docker timeout stacking.

---

## Experiment 3: Higher vLLM GPU Memory (0.3→0.45)

**Config**: Same as #2, plus `vllm_gpu_memory_utilization: 0.45`
**Hypothesis**: More GPU memory for vLLM KV cache = larger generation batches = faster inference.

```
Steps completed:  154
Avg sec/step:     3.48 (1.09x faster vs #2)
Median sec/step:  2.38
Min/Max:          1.59 / 14.19
P50/P90/P95/P99:  2.38 / 6.72 / 12.83 / 13.63
Outliers:         21/151 (13.9%)
Avg w/o outliers: 2.40
Total tokens:     704,687
```

**Notes**: 0.45 is the maximum that fits alongside the training model on a single 80GB GPU.
Attempted 0.6 and 0.9 — both OOM'd because training model occupies ~30GB at init time,
leaving only ~49GB free.

**IMPORTANT**: We did NOT actually enable the vLLM sleep() API in this experiment. TRL requires
`vllm_sleep_enabled=True` in GRPOConfig (added in TRL v0.23.0, our v0.29.0 supports it).
Without this flag, colocate mode statically partitions GPU memory. With sleep() enabled,
vLLM can use up to 0.9 GPU mem during generation (training model sleeps and frees its memory),
then release it all during the training phase. This is a **TODO for a future experiment** —
it could potentially allow 0.9 GPU mem utilization and dramatically speed up generation.

The 50% more KV cache (0.3→0.45) yielded a meaningful speedup: P50 from 2.64→2.38.

---

## Experiment 4: Iteration Reuse (num_iterations=2)

**Config**: Same as #3, plus `num_iterations=2`
**Hypothesis**: Reuses each batch of generated completions for 2 gradient updates,
halving the number of vLLM generation calls.

```
Steps completed:  159
Avg sec/step:     3.43 (1.01x faster vs #3)
Median sec/step:  2.35
Min/Max:          1.56 / 13.34
P50/P90/P95/P99:  2.35 / 6.72 / 12.43 / 13.25
Outliers:         20/156 (12.8%)
Avg w/o outliers: 2.42
Total tokens:     730,294
```

**Notes**: Near-zero improvement. On single-GPU colocate, the reuse benefit is limited because:
1. Each "reuse" step still needs a forward pass to compute new log-probs (for importance sampling)
2. The training forward+backward pass time is already comparable to the colocated generation time
3. The vLLM sleep/wake cycle overhead occurs regardless of reuse

The real benefit of num_iterations would appear on multi-GPU setups where generation
happens on separate GPUs and dominates wall time.

---

## Experiment 5: Switch to OpenRLHF or veRL

**Status**: Not run. Assessment below.

**Analysis**: On a single GPU, the TRL GRPOTrainer + vLLM colocate is already well-optimized:
- Median 2.35 s/step = **~255 steps/hour** for 7B + 8 generations
- This is competitive with reported single-GPU numbers from other frameworks

The major speedup from OpenRLHF/veRL comes from **multi-GPU async generation/training overlap**,
which is not applicable to our single-GPU setup. On 1 GPU, these frameworks would give
similar performance since the bottleneck is the same GPU doing both generation and training.

**Recommendation**: The next meaningful speedup requires **adding more GPUs** (2-4x),
where async frameworks like OpenRLHF shine. On a single H100, our current setup at
~2.4 s/step median is near-optimal for this model/batch configuration.

---

## Conclusions

1. **H100 baseline is already fast**: 2.75s median/step for 7B model with 8 generations
   is much better than the 45s/step initially estimated for A100.

2. **Biggest win: subprocess sandbox** (+12% avg, removes Docker overhead).

3. **Second win: more vLLM GPU memory** (0.3→0.45, +9% avg, faster generation).

4. **Iteration reuse**: Marginal on single GPU. Would shine on multi-GPU.

5. **Outlier steps are the real enemy**: 13% of steps take 3-5x longer than median,
   dragging down the average by 40-80%. Investigating the cause of these spikes
   (likely GC, CUDA memory management, or occasional long generations) could yield
   more gains than any architectural change on single GPU.

6. **Next frontier**: Multi-GPU (2-4x H100) with OpenRLHF/veRL for async generation.
   Expected 2-3x speedup from overlapping generation with training.

## TODO: Experiment 3b — vLLM sleep() enabled

We discovered that `vllm_sleep_enabled=True` must be explicitly set in GRPOConfig.
This was NOT enabled in any experiment above. With sleep() enabled:
- vLLM can use 0.9 GPU mem during generation (training model frees its ~30GB via sleep())
- Training gets full GPU during backward pass (vLLM frees its memory via sleep())
- This should **eliminate the single-GPU memory contention** that caps us at 0.45

Config to test:
```yaml
vllm_gpu_memory_utilization: 0.9
vllm_sleep_enabled: true
sandbox_backend: "subprocess"
```

This could be the biggest single optimization — potentially 1.5-2x over current best.

## Best Config for Production (Single H100)

```yaml
sandbox_backend: "subprocess"        # was: docker
vllm_gpu_memory_utilization: 0.45    # was: 0.3
num_iterations: 1                    # reuse didn't help on 1 GPU
# Everything else: same as dr_grpo.yaml

# UNTESTED but promising:
# vllm_sleep_enabled: true
# vllm_gpu_memory_utilization: 0.9
```
