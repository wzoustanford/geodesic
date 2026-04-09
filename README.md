![alt text](imgs/rl_sys_uml.png)

## Geodesic: Open-source Reinforcement Learning Framework 
Geodesic: shortest path to deployment for reinforcement learning 

Applications: Robotics, VLA, LLM Post-training, RL research 

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and install dependencies:
```bash
git clone https://github.com/wzoustanford/geodesic.git 
cd geodesic
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Python 3.12 is required — mujoco and metaworld do not yet have wheels for newer versions. 

## Metaworld MT10 
python -m tests.ut_online_rl_metaworld_mt10 

## Offline RL 
# prepare/generate your dataset (sample_data_oviss.csv), with 18 features and 1 action defined in configs/vaso_single_action_data_config, in trajectory sequences 
# the data will be offline trajetories already collected, and will be split into train/val/test 
python -m tests.ut_offline_rl 

## RLDataset Schema 

| Field | Type | Shape | Description |
|---|---|---|---|
| `states` | `np.ndarray` | `(N, S)` | Normalized observed states for each transition |
| `actions` | `np.ndarray` | `(N, A)` | Actions taken at each transition |
| `rewards` | `np.ndarray` | `(N,)` | Scalar reward received after each transition |
| `next_states` | `np.ndarray` | `(N, S)` | Normalized successor states after each transition |
| `dones` | `np.ndarray` | `(N,)` | Episode termination flags (`True` if terminal) |
| `n_transitions` | `int` | `—` | Total number of transitions `N` across all trajectories |
| `n_trajs` | `int` | `—` | Total number of trajectories collected |
| `state_features` | `list[str]` | `(S,)` | Ordered list of feature names corresponding to state dimensions |

> **Shape key:** `N` = number of transitions, `S` = state dimension, `A` = action dimension.
