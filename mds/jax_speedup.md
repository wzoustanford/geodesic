# JAX Speedup Experiments

## Setup

- **Task:** Online RL on MetaWorld MT10 (10-task multi-task robotics)
- **Algorithm:** SAC (Soft Actor-Critic)
- **Model architecture:** Actor: MLP(400, 400, 400) → 8D (mean + log_std). Critic: ConcatQNetwork MLP(400, 400, 400) → 1D (state + action concatenated)
- **Training:** 5 epochs, 500 steps/epoch, 2500 total steps, warmstart 100 steps, batch_size 32
- **Hardware:** MacBook Pro (CPU only, no GPU)

## JAX optimizations implemented

1. **@jax.jit on `_update_pure`** — full SAC update (critic + actor + alpha + soft target update) compiled into a single XLA program
2. **@jax.jit on `_select_actions`** — action sampling compiled
3. **optax training loop** — `optax.adam` + `TrainState.apply_gradients` replaces PyTorch `optimizer.zero_grad(); loss.backward(); optimizer.step()`
4. **TrainState / CriticTrainState** — immutable pytree bundles (params + optimizer state + apply_fn), enabling functional updates inside JIT
5. **distrax distributions** — TanhMultivariateNormalDiag handles tanh squashing + log_prob correction inside JIT graph
6. **jax.vmap over tasks** — per-task critic/actor loss and gradients computed in parallel, then averaged with `jax.tree.map`
7. **jax.tree.map for gradient averaging** — `jax.tree.map(lambda x: x.mean(axis=0), grads)` averages per-task gradients before `apply_gradients`

## Results

### Experiment 1: Baseline (5 epochs, 2500 steps, CPU)

Raw output:

```
# PyTorch (ut_online_rl_metaworld_mt10)
Epoch 1/5, global_step=500, sequences=500, time=0.1min
Epoch 2/5, global_step=1000, sequences=1000, time=0.3min
Epoch 3/5, global_step=1500, sequences=1500, time=0.4min
Epoch 4/5, global_step=2000, sequences=2000, time=0.6min
Epoch 5/5, global_step=2500, sequences=2500, time=0.8min
Online training complete in 0.8min

# JAX (ut_online_rl_metaworld_mt10_jax)
Epoch 1/5, global_step=500, sequences=500, time=0.3min
Epoch 2/5, global_step=1000, sequences=1000, time=0.4min
Epoch 3/5, global_step=1500, sequences=1500, time=0.5min
Epoch 4/5, global_step=2000, sequences=2000, time=0.5min
Epoch 5/5, global_step=2500, sequences=2500, time=0.6min
Online training complete in 0.6min
```

| Framework | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Total |
|---|---|---|---|---|---|---|
| PyTorch | 0.1min | 0.3min | 0.4min | 0.6min | 0.8min | **0.8min** |
| JAX | 0.3min | 0.4min | 0.5min | 0.5min | 0.6min | **0.6min** |

**Observations:**
- JAX total time: **0.6min** vs PyTorch **0.8min** — **25% speedup**
- JAX epoch 1 is slower (0.3min vs 0.1min) due to JIT compilation overhead on first call
- JAX epochs 2-5 are consistently faster — compiled code executes without Python interpreter overhead
- Speedup expected to be larger with GPU and larger batch sizes where XLA kernel fusion has more impact

### Experiment 2: After adding jax.vmap + jax.tree.map (5 epochs, 2500 steps, CPU)

Added `jax.vmap` over tasks for critic and actor loss/gradient computation. Data reshaped from flat `(N, feat)` to `(num_tasks, per_task_batch, feat)` before vmapped functions. Per-task gradients averaged with `jax.tree.map(lambda x: x.mean(axis=0), grads)`.

Raw output:

```
# JAX + vmap (ut_online_rl_metaworld_mt10_jax)
Epoch 1/5, global_step=500, sequences=500, time=0.3min
Epoch 2/5, global_step=1000, sequences=1000, time=0.4min
Epoch 3/5, global_step=1500, sequences=1500, time=0.5min
Epoch 4/5, global_step=2000, sequences=2000, time=0.6min
Epoch 5/5, global_step=2500, sequences=2500, time=0.7min
Online training complete in 0.7min
```

| Framework | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Total |
|---|---|---|---|---|---|---|
| PyTorch | 0.1min | 0.3min | 0.4min | 0.6min | 0.8min | **0.8min** |
| JAX (Exp 1: jit only) | 0.3min | 0.4min | 0.5min | 0.5min | 0.6min | **0.6min** |
| JAX (Exp 2: jit + vmap) | 0.3min | 0.4min | 0.5min | 0.6min | 0.7min | **0.7min** |

**Observations:**
- JAX + vmap: **0.7min** vs JAX jit-only **0.6min** — vmap added ~0.1min overhead
- The overhead is expected on CPU with small batch sizes: vmap's benefit is parallelism across tasks, but on CPU the tasks are executed sequentially anyway
- On GPU, vmap would execute all 10 tasks simultaneously as a single batched operation — expect significant speedup there
- The vmap version is **algorithmically correct** for multi-task RL (per-task gradients averaged), while the jit-only version treated all tasks as one flat batch
