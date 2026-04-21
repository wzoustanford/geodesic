import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")

import ray
import numpy as np
from jax_agents import JAXSACAgent
from orchestrator import ParallelReplayOrchestrator
from envs import MetaworldConfig


class MT10ParallelReplayConfig:
    RL_MODE = "ONLINE"
    SEQ_LEN = 1
    STRIDE = 1
    BUFFER_CAPACITY = 100000
    TRAIN_BATCH_SIZE = 32
    SHUFFLE = False
    NUM_STEPS_PER_EPOCH = 200
    SEED = 42


def ut_parallel_replay_mt10_jax():
    ray.init(num_cpus=4, ignore_reinit_error=True)

    env_config = MetaworldConfig(env_id="MT10", terminate_on_success=False)
    state_dim = env_config.observation_space.shape[0]   # 59 (49 state + 10 one-hot)
    action_dim = int(np.prod(env_config.action_space.shape))  # 4
    agent = JAXSACAgent(state_dim=state_dim, action_dim=action_dim)
    data_config = MT10ParallelReplayConfig()

    orc = ParallelReplayOrchestrator(
        agent=agent,
        data_config=data_config,
        num_epochs=3,
        env_config=env_config,
        warmstart_steps=50,
        num_workers=3,
        worker_weight_ckpt_interval=50,
        queue_maxsize=500,
        drain_interval=0.5,
        drain_threshold=10,
    )
    orc.start_online()

    ray.shutdown()
    print("ut_parallel_replay_mt10_jax passed")


if __name__ == "__main__":
    ut_parallel_replay_mt10_jax()

"""
(geodesic) willzou@Wills-MacBook-Pro-3 geodesic % python3 -m tests.ut_parallel_replay_metaworld_mt10_jax
2026-04-18 17:58:32,891	INFO worker.py:2012 -- Started a local Ray instance.
/Users/willzou/code/geodesic/.venv/lib/python3.12/site-packages/ray/_private/worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
----------starting parallel replay with online RL training----------
Workers: 3, worker_weight_ckpt_interval: 50
Collecting warmstart transitions...
(DataWorker pid=29318) /Users/willzou/code/geodesic/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:236: UserWarning: WARN: Box low's precision lowered by casting to float32, current low.dtype=float64
(DataWorker pid=29318)   gym.logger.warn(
(DataWorker pid=29318) /Users/willzou/code/geodesic/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:306: UserWarning: WARN: Box high's precision lowered by casting to float32, current high.dtype=float64
(DataWorker pid=29318)   gym.logger.warn(
Warmstart complete. Buffer size: 78. Beginning training...
Epoch 1/3, global_step=200, buffer=1813, time=0.9min
Epoch 2/3, global_step=400, buffer=2812, time=1.3min
Epoch 3/3, global_step=600, buffer=3607, time=1.6min
Parallel training complete in 1.6min
(DataWorker pid=29320) /Users/willzou/code/geodesic/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:236: UserWarning: WARN: Box low's precision lowered by casting to float32, current low.dtype=float64 [repeated 32x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(DataWorker pid=29320)   gym.logger.warn( [repeated 64x across cluster]
(DataWorker pid=29320) /Users/willzou/code/geodesic/.venv/lib/python3.12/site-packages/gymnasium/spaces/box.py:306: UserWarning: WARN: Box high's precision lowered by casting to float32, current high.dtype=float64 [repeated 32x across cluster]
ut_parallel_replay_mt10_jax passed
"""
