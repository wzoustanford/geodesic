# JAX Implementation Plan: SAC on MetaWorld

Reference: metaworld-algorithms-grin (JAX/Flax/optax)
Target: geodesic (currently PyTorch, adding JAX path)

## A. Network Definitions (nn.Module)

| # | Reference (JAX/Flax) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 1 | `nn/base.py:11` — MLP (Flax nn.Module) | `models.py` — ConcatQNetwork (torch nn.Module) | Rewrite as Flax nn.Module |
| 2 | `rl/networks.py:22` — ContinuousActionPolicyTorso | `agents.py` — `_make_actor` builds nn.Sequential | Rewrite as Flax nn.Module |
| 3 | `rl/networks.py:68` — ContinuousActionPolicy | `agents.py` — `_get_action_dist` | Flax module returning distrax distribution |
| 4 | `rl/networks.py:211` — QValueFunction | `models.py` — ConcatQNetwork | Rewrite as Flax nn.Module |
| 5 | `nn/distributions.py:11` — TanhMultivariateNormalDiag | `agents.py` — manual tanh + log_prob correction | Use distrax TanhMultivariateNormalDiag |
| 6 | `sac.py:45` — Temperature (log_alpha module) | `agents.py` — `self.log_alpha` (bare tensor) | Flax nn.Module with learnable param |

## B. Training State & Optimizers (TrainState + optax)

| # | Reference (JAX) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 7 | `sac.py:131` — TrainState.create for actor | `agents.py` — `self.actor` + `self.actor_optimizer` | Replace with TrainState + optax.adam |
| 8 | `sac.py:143` — CriticTrainState.create (with target_params) | `agents.py` — `self.q1/q2` + `self.q1_target/q2_target` + optimizers | Replace with CriticTrainState + optax |
| 9 | `sac.py:151` — TrainState.create for alpha | `agents.py` — `self.log_alpha` + `self.alpha_optimizer` | Replace with TrainState + optax |
| 10 | `sac.py:296` — `optax.incremental_update` for target nets | `agents.py` — `_soft_update_targets` (manual Polyak) | Replace with optax.incremental_update |

## C. JIT-Compiled Functions (@jax.jit)

| # | Reference (JAX) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 11 | `sac.py:62` — `_sample_action` | `agents.py` — `sample_action` | Add @jax.jit |
| 12 | `sac.py:72` — `_eval_action` | Not implemented | Add @jax.jit eval function |
| 13 | `sac.py:189` — `_update_inner` | `agents.py` — `update()` | Rewrite as @jax.jit with functional style |

## D. Gradient Computation (jax.value_and_grad)

| # | Reference (JAX) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 14 | `sac.py:222` — critic loss + grad | `agents.py` — `critic_loss` + `critic_step` (backward/step) | Replace with jax.value_and_grad + apply_gradients |
| 15 | `sac.py:245` — alpha loss + grad | `agents.py` — manual backward + alpha_optimizer.step | Replace with jax.value_and_grad + apply_gradients |
| 16 | `sac.py:280` — actor loss + grad | `agents.py` — manual backward + actor_optimizer.step | Replace with jax.value_and_grad + apply_gradients |

## E. Gradient Stopping (jax.lax.stop_gradient)

| # | Reference (JAX) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 17 | `sac.py:214` — stop_gradient on target Q | `agents.py` — `torch.no_grad()` context | Replace with jax.lax.stop_gradient |
| 18 | `sac.py:269` — stop_gradient on alpha before critic | `agents.py` — `.detach()` | Replace with jax.lax.stop_gradient |

## F. RNG Key Management (jax.random)

| # | Reference (JAX) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 19 | `sac.py:120` — PRNGKey init | Not present (PyTorch uses global RNG) | Add explicit key management |
| 20 | `sac.py:66,192` — jax.random.split | Not present | Add key splitting at each update/sample |

## G. Multi-task Extensions (jax.vmap + jax.tree.map)

| # | Reference (JAX) | Geodesic (Current PyTorch) | Change needed |
|---|---|---|---|
| 21 | `mtsac.py:239` — tree.map to group data by task | orchestrator.py — flatten batch reshape | Add tree.map for task grouping |
| 22 | `mtsac.py:319` — vmap critic loss over tasks | Not present | Add vmap for per-task critic gradients |
| 23 | `mtsac.py:401` — vmap actor loss over tasks | Not present | Add vmap for per-task actor gradients |
| 24 | `mtsac.py:332,407` — tree.map to average gradients | Not present | Add tree.map for gradient averaging |

## Proposed Module Structure

| Module | Responsibility | Items from table |
|---|---|---|
| `jax_models.py` | Flax nn.Module definitions (MLP, policy, Q-network, Temperature) | #1-6 |
| `jax_agents.py` | JAX SAC agent (TrainState, update, sample) | #7-20 |
| `jax_agents.py` (MTSAC subclass) | Multi-task extensions with vmap/tree.map | #21-24 |

## Key Architectural Shift

PyTorch is **stateful**: model holds weights, optimizer holds state, `.backward()` accumulates gradients.

JAX is **functional**: TrainState is an immutable pytree, `_update_inner` returns a new state. The update pattern becomes:

```
self = self.replace(actor=new_actor, critic=new_critic, alpha=new_alpha)
```

PyTorch equivalents mapped to JAX:
- `model.parameters()` → `train_state.params`
- `optimizer.zero_grad(); loss.backward(); optimizer.step()` → `jax.value_and_grad(loss_fn)(params)` + `train_state.apply_gradients(grads=grads)`
- `torch.no_grad()` / `.detach()` → `jax.lax.stop_gradient`
- `model.to(device)` → `jax.device_put` (usually automatic)
- Global RNG → Explicit `jax.random.PRNGKey` + `jax.random.split`

## Orchestrator Changes

The orchestrator (`orchestrator.py`) currently has three PyTorch-specific points:

1. **Line 1:** `import torch`
2. **Line 96-97:** `torch.is_tensor(actions)` / `actions.cpu().numpy()` — tensor-to-numpy conversion
3. **Line 121-124:** `b.dim()` and `b.reshape()` — torch tensor methods on DataLoader output

To make the orchestrator framework-agnostic, these should be replaced with numpy
operations (e.g. `np.asarray(actions)` works for torch tensors, JAX arrays, and numpy).
The batch reshape can use `np.ndim()` and `np.reshape()`.

### Data Transfer: DataLoader → JAX

**CPU torch tensor → JAX array:**
`jnp.array(tensor.numpy())` — numpy view is zero-copy, then one copy to JAX device. Cheap.

**GPU torch tensor → JAX array (same GPU):**
No direct zero-copy path. Would require GPU → CPU → GPU round trip, unless using
`jax.dlpack` / `torch.utils.dlpack` for zero-copy sharing (fragile).

**For our case:** DataLoader runs on CPU by default, so tensors are CPU torch tensors.
The conversion `jnp.array(tensor.numpy())` is cheap.

### Design Decision: Keep DataLoader, Convert in JAX Agent

Keep the PyTorch DataLoader which operates on CPU RAM. Convert the batch from torch
tensor to jax.array inside `jax_agents.py`, including moving data to GPU memory.

We use DataLoader due to important design choices: lazy loading and parallelization
when data gets large and needs SSD or hard disk storage. For now, keep this
implementation. We could experiment with a native numpy replay buffer later for JAX,
if the data transfer is a bottleneck.
