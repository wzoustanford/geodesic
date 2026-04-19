# EfficientZero Atari Environment Integration Investigation

## Goal

Integrate EfficientZero's Atari environment, parallel replay buffer, and Ray support into geodesic. Use a DQN algorithm (JAX) for the model, adding methods in `jax_agents.py` and models in `jax_models.py`.

## EfficientZero Atari Environment Architecture

### Wrapper Stack

The environment is built as a chain of wrappers, each adding one preprocessing step:

```
gym.make("SpaceInvadersNoFrameskip-v4")       # raw Atari env via gym==0.15.7
  -> NoopResetEnv(noop_max=30)                 # 1-30 random no-ops on reset for stochastic starts
    -> MaxAndSkipEnv(skip=4)                   # repeat action 4 frames, max-pool last 2 frames
      -> TimeLimit(max_episode_steps)          # hard episode length cap
        -> EpisodicLifeEnv                     # life loss = done (train only)
          -> WarpFrame(96x96, grayscale)       # resize + grayscale via cv2
            -> AtariWrapper(cvt_string=True)   # uint8 cast, optional JPEG encoding for memory
```

### How the Environment is Created and Used

`AtariConfig` (inherits `BaseConfig`) acts as a **factory**. It never stores an env instance.

- `config.new_game(seed, test, ...)` builds a fresh wrapped env each time it's called
- `config.set_game(env_name)` calls `new_game()` once just to discover `action_space_size`, then discards the env
- The env name (e.g. `"SpaceInvadersNoFrameskip-v4"`) flows from CLI `--env` arg -> `set_config(args)` -> `set_game(args.env)` -> stored as `self.env_name`

Callers of `new_game()`:
1. `selfplay_worker.py:112` - `DataWorker.run()` creates `p_mcts_num` (default 4) parallel envs for self-play
2. `test.py:79` - creates envs for evaluation
3. `config/atari/__init__.py:104` - one-shot to discover `action_space_size`

`env.step(action)` is called in `selfplay_worker.py:299` inside the `DataWorker.run()` loop.

### Key Source Files (Reference: `/Users/willzou/code/EfficientZero/`)

| File | Purpose |
|------|---------|
| `config/atari/__init__.py` | `AtariConfig(BaseConfig)` - Atari hyperparams, `new_game()` factory, `set_game()` |
| `config/atari/env_wrapper.py` | `AtariWrapper(Game)` - final wrapper, uint8 cast, JPEG encoding |
| `core/config.py` | `BaseConfig` - all hyperparameters, abstract methods, scalar transforms |
| `core/game.py` | `Game` base class (env holder), `GameHistory` (trajectory block storage) |
| `core/utils.py` | `make_atari()`, `NoopResetEnv`, `MaxAndSkipEnv`, `EpisodicLifeEnv`, `WarpFrame` |
| `core/selfplay_worker.py` | `DataWorker` - Ray actor that runs self-play, calls `env.step()` |
| `core/replay_buffer.py` | `ReplayBuffer` - Ray actor, prioritized experience replay |
| `core/storage.py` | `SharedStorage`, `QueueStorage` - Ray actors for model weights and batch queues |
| `core/train.py` | Training loop orchestration, creates all Ray actors |
| `main.py` | CLI entry point, parses `--env`, calls `set_config()` then `train()` |

### BaseConfig Key Parameters (Atari Overrides)

```
training_steps=100000     frame_skip=4              discount=0.997 (then **= frame_skip)
max_moves=12000           stacked_observations=4    clip_reward=True
history_length=400        batch_size=256            episode_life=True
cvt_string=True           image_based=True          num_actors=1
td_steps=5                num_simulations=50        lr_init=0.2
```

Network architecture (set in AtariConfig post-init):
```
blocks=1                  channels=64 (32 if grayscale)    downsample=True
reduced_channels_reward=16    reduced_channels_value=16    reduced_channels_policy=16
resnet_fc_reward_layers=[32]  resnet_fc_value_layers=[32]  resnet_fc_policy_layers=[32]
lstm_hidden_size=512          lstm_horizon_len=5
```

### Wrapper Details

**NoopResetEnv**: On `reset()`, takes 1-30 random no-op actions (action 0) to create stochastic initial states. `step()` is pass-through.

**MaxAndSkipEnv**: Repeats each action for `skip` frames (default 4). Accumulates reward. Max-pools the last 2 frames to avoid Atari sprite flickering. Stores a `(2, H, W, C)` observation buffer.

**EpisodicLifeEnv**: Checks `env.unwrapped.ale.lives()` after each step. If lives decreased and lives > 0, sets `done=True`. On `reset()`, only truly resets when `was_real_done=True`; otherwise does a no-op step to advance past the death frame. Used during training only (not test).

**WarpFrame**: `gym.ObservationWrapper`. Resizes frames to `width x height` (96x96 in EfficientZero) using `cv2.INTER_AREA`. Optionally converts RGB to grayscale via `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)`. Output shape: `(H, W, 1)` grayscale or `(H, W, 3)` color.

**AtariWrapper**: Inherits `Game` base class (stores `env`, `action_space_size`, `discount`). Casts observations to `np.uint8`. Optionally JPEG-encodes observations via `cv2.imencode('.jpg', arr)` to save replay buffer memory. Provides `legal_actions()` returning all valid action indices.

### Ray Parallel Architecture

```
DataWorker (Ray actor, 0.125 GPU each)
  -> creates p_mcts_num=4 envs via config.new_game()
  -> runs self-play loop: obs -> model inference -> MCTS -> env.step()
  -> stores trajectories in trajectory_pool
  -> sends to ReplayBuffer via replay_buffer.save_pools.remote()

ReplayBuffer (Ray actor)
  -> stores GameHistory blocks with priorities
  -> prioritized sampling (alpha=0.6, beta=0.4)
  -> capacity: 25M transitions

BatchWorker_CPU (Ray actor, 14x)
  -> samples from ReplayBuffer
  -> prepares reward/value/policy targets

BatchWorker_GPU (Ray actor, 0.125 GPU each, 20x)
  -> computes targets via model inference
  -> pushes to batch_storage queue

Training (main process)
  -> pops from batch_storage
  -> forward + backward + optimizer step
  -> updates SharedStorage model weights
```

### GameHistory (Trajectory Storage)

Long Atari episodes are split into blocks of `history_length=400` steps. Each `GameHistory` stores:
- `obs_history`: observations (as JPEG strings if `cvt_string=True`)
- `actions`: action indices
- `rewards`: clipped rewards
- `child_visits`: MCTS visit count distributions
- `root_values`: MCTS root values

When a block is full (`is_full()`), it's saved as `last_game_history` and a new block starts with overlapping frame stack. `pad_over()` connects consecutive blocks for correct bootstrapped value targets. `game_over()` converts arrays to numpy and moves `obs_history` to Ray object store via `ray.put()`.

## Version Incompatibility

### EfficientZero Requirements

```
gym[atari,roms,accept-rom-license]==0.15.7
numpy==1.19.5
ray==1.0.0
cython==0.29.23
opencv-python==4.5.1.48
kornia==0.6.6
```

- `gym==0.15.7`: `env.step()` returns `(obs, reward, done, info)` (4 values), `env.reset()` returns `obs` only
- `numpy==1.19.5`: wheels for Python 3.6-3.9 only
- Effective Python requirement: **3.6-3.9**

### Geodesic Requirements

```
requires-python = ">=3.12"
gymnasium>=1.1.1
numpy>=2.4.0
torch>=2.10.0
jax>=0.9.2
flax>=0.12.6
```

- `gymnasium`: `env.step()` returns `(obs, reward, terminated, truncated, info)` (5 values), `env.reset()` returns `(obs, info)`
- Effective Python requirement: **>=3.12**

### Incompatibilities

| Dependency | EfficientZero | Geodesic | Conflict |
|-----------|---------------|----------|----------|
| Python | 3.6-3.9 | >=3.12 | Cannot share venv |
| gym/gymnasium | gym==0.15.7 | gymnasium>=1.1.1 | Different packages, different step/reset API |
| numpy | 1.19.5 | >=2.4.0 | Major version incompatibility |

## Proposed Solution: Separate venv

Create a dedicated venv for Atari that supports both the gym==0.15.7 Atari environment and JAX for the DQN model.

### Requirements for the Atari venv

Must support:
- `gym==0.15.7` with Atari ROMs
- JAX + Flax + Optax (for DQN model)
- Ray (for parallel replay)
- numpy (version compatible with both gym and JAX)
- opencv-python (for WarpFrame, JPEG encoding)

### Constraints to Resolve

1. **Python version**: Need a version that works with both `gym==0.15.7` (<=3.9) and JAX. Modern JAX (>=0.4.x) requires Python >=3.9. This narrows to **Python 3.9**.
2. **numpy version**: `gym==0.15.7` works with numpy 1.x. JAX on Python 3.9 should work with numpy 1.x as well (e.g. numpy 1.24.x or 1.26.x). Need to test exact compatible version.
3. **JAX version**: Latest JAX supporting Python 3.9 (JAX dropped 3.9 in 0.4.31). Need to find the last compatible version.
4. **Ray version**: `ray==1.0.0` is very old. May need a newer version that still supports Python 3.9.

### Proposed venv Setup

```bash
uv venv --python 3.9 .venv-atari
source .venv-atari/bin/activate

# Core
uv pip install gym[atari,accept-rom-license]==0.15.7
uv pip install numpy==1.24.4  # last 1.x that works with both gym and JAX

# JAX ecosystem (find last versions supporting Python 3.9)
uv pip install jax jaxlib flax optax

# Parallel replay
uv pip install ray

# Image processing
uv pip install opencv-python

# Geodesic (install as editable, may need to relax version constraints)
uv pip install -e . --no-deps
```

### Open Questions

- Exact JAX version ceiling for Python 3.9?
- Can Ray 1.x coexist with modern JAX, or do we need a newer Ray?
- Should geodesic's Atari code live in the main repo and be importable from both venvs, or in a separate package?
- Alternative: use gymnasium + ale-py in the main geodesic venv and adapt the 4-return API to 5-return? This avoids the separate venv but changes the env interaction code.

## Next Steps

1. Determine exact compatible versions (JAX, Ray, numpy) for Python 3.9
2. Create and test the Atari venv
3. Implement the Atari wrappers in `envs/atari.py` (exact replica of reference code)
4. Implement parallel replay buffer with Ray
5. Implement DQN agent in `jax_agents.py` and `jax_models.py`
