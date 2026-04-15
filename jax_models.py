import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState


class Temperature(nn.Module):
    """Learnable temperature (entropy coefficient) for SAC."""
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full((1,), jnp.log(self.initial_temperature)),
        )

    def __call__(self):
        return jnp.exp(self.log_alpha)


class MultiTaskTemperature(nn.Module):
    """Per-task learnable temperature for multi-task SAC."""
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(self, task_ids):
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))


class CriticTrainState(TrainState):
    """TrainState with target network parameters for soft updates."""
    target_params: dict


class ActorNetwork(nn.Module):
    """Policy network: outputs mean + log_std for each action dimension."""
    hidden_dim: int
    depth: int
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim * 2)(x)


class JAXConcatQNetwork(nn.Module):
    """Q-network that concatenates state and action before the MLP."""
    hidden_dim: int
    depth: int

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)
