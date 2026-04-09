![alt text](imgs/rl_sys_uml.png)

## Geodesic: Open-source Reinforcement Learning Framework 
Geodesic: shortest path to production deployment of reinforcement learning 

Applications: Robotics, VLA, LLM Post-training, RL research 

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and install dependencies:
```bash
git clone <repo-url>
cd geodesic
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Python 3.12 is required — mujoco and metaworld do not yet have wheels for newer versions.

