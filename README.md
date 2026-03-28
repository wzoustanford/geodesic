![alt text](imgs/rl_sys_uml.png)

## Geodesic: Open-source Reinforcement Learning Framework 
Geodesic: shortest path to production deployment of reinforcement learning 

Applications: Robotics, VLA, LLM Post-training, RL research 

Replay buffer Schema: 
## Dataset Schema

| Field | Type | Shape | Description |
| `states` | `np.ndarray` | `(N, S)` | Normalized observed states for each transition |
| `actions` | `np.ndarray` | `(N, A)` | Actions taken at each transition |
| `rewards` | `np.ndarray` | `(N,)` | Scalar reward received after each transition |
| `next_states` | `np.ndarray` | `(N, S)` | Normalized successor states after each transition |
| `dones` | `np.ndarray` | `(N,)` | Episode termination flags (`True` if terminal) |
| `n_transitions` | `int` | `—` | Total number of transitions `N` across all trajectories |
| `n_trajs` | `int` | `—` | Total number of trajectories collected |
| `state_features` | `list[str]` | `(S,)` | Ordered list of feature names corresponding to state dimensions |

> **Shape key:** `N` = number of transitions, `S` = state dimension, `A` = action dimension.
