import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")

from agents import SACAgent
from jax_agents import JAXSACAgent
from orchestrator import Orchestrator
from envs import MetaworldConfig
import numpy as np
import os

class MT10OnlineConfig:
    RL_MODE = "ONLINE"
    SEQ_LEN = 1
    STRIDE = 1
    BUFFER_CAPACITY = 100000
    TRAIN_BATCH_SIZE = 32
    SHUFFLE = False
    NUM_STEPS_PER_EPOCH = 500
    SEED = 42

SAVE_DIR = './checkpoints/ut_save_load'

def make_env_config():
    return MetaworldConfig(env_id="MT10", terminate_on_success=False)

def get_dims(env_config):
    state_dim = env_config.observation_space.shape[0]
    action_dim = int(np.prod(env_config.action_space.shape))
    return state_dim, action_dim

# ============================================================================
# (1) PyTorch: train 2 epochs, save
# ============================================================================
def test_pytorch_train():
    print("\n" + "="*60)
    print("(1) PyTorch: train 2 epochs")
    print("="*60)
    env_config = make_env_config()
    state_dim, action_dim = get_dims(env_config)
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    data_config = MT10OnlineConfig()
    orc = Orchestrator(
        agent=agent, data_config=data_config,
        num_epochs=2, env_config=env_config, warmstart_steps=100,
    )
    orc.start_online()
    os.makedirs(SAVE_DIR, exist_ok=True)
    agent.save(f"{SAVE_DIR}/pytorch_epoch2.pt")
    return agent

# ============================================================================
# (2) PyTorch: load and train 2 more epochs
# ============================================================================
def test_pytorch_resume():
    print("\n" + "="*60)
    print("(2) PyTorch: load and train 2 more epochs")
    print("="*60)
    env_config = make_env_config()
    state_dim, action_dim = get_dims(env_config)
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(f"{SAVE_DIR}/pytorch_epoch2.pt")
    data_config = MT10OnlineConfig()
    orc = Orchestrator(
        agent=agent, data_config=data_config,
        num_epochs=2, env_config=env_config, warmstart_steps=0,
    )
    orc.start_online()
    agent.save(f"{SAVE_DIR}/pytorch_epoch4.pt")
    return agent

# ============================================================================
# (3) JAX: train 2 epochs, save
# ============================================================================
def test_jax_train():
    print("\n" + "="*60)
    print("(3) JAX: train 2 epochs")
    print("="*60)
    env_config = make_env_config()
    state_dim, action_dim = get_dims(env_config)
    agent = JAXSACAgent(state_dim=state_dim, action_dim=action_dim)
    data_config = MT10OnlineConfig()
    orc = Orchestrator(
        agent=agent, data_config=data_config,
        num_epochs=2, env_config=env_config, warmstart_steps=100,
    )
    orc.start_online()
    os.makedirs(SAVE_DIR, exist_ok=True)
    agent.save(f"{SAVE_DIR}/jax_epoch2", epoch=2, data_config=data_config)
    return agent

# ============================================================================
# (4) JAX: load and train 2 more epochs
# ============================================================================
def test_jax_resume():
    print("\n" + "="*60)
    print("(4) JAX: load and train 2 more epochs")
    print("="*60)
    env_config = make_env_config()
    state_dim, action_dim = get_dims(env_config)
    agent = JAXSACAgent(state_dim=state_dim, action_dim=action_dim)
    epoch, data_config = agent.load(f"{SAVE_DIR}/jax_epoch2")
    print(f"Resumed from epoch {epoch}")
    data_config = MT10OnlineConfig()
    orc = Orchestrator(
        agent=agent, data_config=data_config,
        num_epochs=2, env_config=env_config, warmstart_steps=0,
    )
    orc.start_online()
    agent.save(f"{SAVE_DIR}/jax_epoch4", epoch=4, data_config=data_config)
    return agent

if __name__ == "__main__":
    #test_pytorch_train()
    #test_pytorch_resume()
    test_jax_train()
    test_jax_resume()
    print("\n" + "="*60)
    print("All 4 save/load tests completed successfully!")
    print("="*60)
