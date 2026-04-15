import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
from agents import SACAgent, Agent
from jax_models import JAXConcatQNetwork, ActorNetwork, Temperature, MultiTaskTemperature, CriticTrainState


## [proposal] framework design protocal, jax accelerated code can be listed on the top section as pure functions 
## class structure is kept at the bottom section, retainin inheritance/abstractions including inheritance from pytorch friendly classes 

def extract_task_weights(alpha_params, task_ids):
    """Compute per-task loss weights from temperature params."""
    log_alpha = alpha_params["params"]["log_alpha"]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)
    task_weights *= log_alpha.shape[0]
    return task_weights


@jax.jit
def _select_actions(actor, states, key):
    """Pure function version of select_actions."""
    output = actor.apply_fn(actor.params, states)
    mean, log_std = jnp.split(output, 2, axis=-1)
    log_std = jnp.clip(log_std, -20, 2)
    dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = distrax.Transformed(dist, distrax.Block(distrax.Tanh(), 1))
    actions, log_probs = dist.sample_and_log_prob(seed=key)
    return actions, log_probs


# [to evaluate] consider taking out jax.vmap and use_task_vmap because there is no
# speedup and it's only useful when we split per task using task weights.
@jax.jit(static_argnames=['num_tasks', 'use_task_vmap'])
def _update_pure(actor, critic, alpha, key,
                 states, actions, rewards, next_states, dones,
                 num_tasks, use_task_vmap, gamma, tau, target_entropy):
    key, critic_key, actor_key = jax.random.split(key, 3)
    alpha_val = alpha.apply_fn(alpha.params)

    if use_task_vmap:
        # Reshape flat (N, feat) → (num_tasks, per_task_batch, feat)
        def reshape_by_task(x):
            return x.reshape(num_tasks, -1, *x.shape[1:])
        s_t, a_t, r_t, ns_t, d_t = map(reshape_by_task,
            [states, actions, rewards, next_states, dones])

    # --- 1. Critic update ---
    if use_task_vmap:
        # Compute target Q per task
        def compute_target(ns, r, d):
            next_actions, next_log_probs = _select_actions(actor, ns, critic_key)
            target_qs = critic.apply_fn(critic.target_params, ns, next_actions)
            min_q = jnp.min(target_qs, axis=0).squeeze(-1)
            return r + gamma * (1 - d) * (min_q - alpha_val * next_log_probs)

        target_q = jax.lax.stop_gradient(
            jax.vmap(compute_target)(ns_t, r_t, d_t)
        )

        # Critic loss per task
        def critic_loss_fn(params, s, a, tq):
            q_pred = critic.apply_fn(params, s, a)
            return ((q_pred.squeeze(-1) - tq) ** 2).mean()

        critic_loss, critic_grads = jax.vmap(
            jax.value_and_grad(critic_loss_fn),
            in_axes=(None, 0, 0, 0),
        )(critic.params, s_t, a_t, target_q)
        critic_loss = critic_loss.mean()
        critic_grads = jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
    else:
        next_actions, next_log_probs = _select_actions(actor, next_states, critic_key)
        target_qs = critic.apply_fn(critic.target_params, next_states, next_actions)
        min_q = jnp.min(target_qs, axis=0).squeeze(-1)
        target_q = rewards + gamma * (1 - dones) * (
            min_q - alpha_val * next_log_probs
        )
        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(params):
            q_pred = critic.apply_fn(params, states, actions)
            return ((q_pred.squeeze(-1) - target_q) ** 2).mean()

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic.params)

    critic = critic.apply_gradients(grads=critic_grads)

    # --- 2. Actor update ---
    if use_task_vmap:
        def actor_loss_fn(actor_params, s):
            a, lp = _select_actions(actor.replace(params=actor_params), s, actor_key)
            q_vals = critic.apply_fn(critic.params, s, a)
            min_q = jnp.min(q_vals, axis=0)
            return (alpha_val * lp - min_q).mean(), lp

        (actor_loss, log_probs), actor_grads = jax.vmap(
            jax.value_and_grad(actor_loss_fn, has_aux=True),
            in_axes=(None, 0),
        )(actor.params, s_t)
        actor_loss = actor_loss.mean()
        log_probs = log_probs.reshape(-1)
        actor_grads = jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
    else:
        def actor_loss_fn(actor_params):
            a, lp = _select_actions(actor.replace(params=actor_params), states, actor_key)
            q_vals = critic.apply_fn(critic.params, states, a)
            min_q = jnp.min(q_vals, axis=0)
            return (alpha_val * lp - min_q).mean(), lp

        (actor_loss, log_probs), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor.params)

    actor = actor.apply_gradients(grads=actor_grads)

    # --- 3. Alpha (temperature) update ---
    def alpha_loss_fn(alpha_params):
        log_alpha = alpha_params["params"]["log_alpha"]
        return (-log_alpha * (log_probs + target_entropy)).mean()

    alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha.params)
    alpha = alpha.apply_gradients(grads=alpha_grads)

    # --- 4. Soft update targets ---
    critic = critic.replace(
        target_params=optax.incremental_update(
            critic.params, critic.target_params, tau
        )
    )

    logs = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'sac_alpha': alpha_val.squeeze(),
    }
    return actor, critic, alpha, key, logs


class JAXSACAgent(SACAgent):
    """
    JAX implementation of SAC, inheriting from SACAgent (PyTorch).
    Overrides all framework-specific methods while reusing:
    _transform_actions, compute_cql_loss, get_save_path, train.
    """

    def __init__(self, state_dim, action_dim, num_tasks=10, use_task_vmap=False,
                 hidden_dim=400, depth=3, lr=3e-3, gamma=0.95, tau=0.005, seed=42):
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.use_task_vmap = use_task_vmap
        self.hidden_dim = hidden_dim
        self.depth = depth
        Agent.__init__(self)  # skip torch init in BaseQLAgent/SACAgent
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.training_step = 0
        self.target_entropy = -action_dim

        # RNG key management (replaces PyTorch global RNG)
        key = jax.random.PRNGKey(seed)
        key, actor_key, critic_key, alpha_key = jax.random.split(key, 4)
        self.key = key

        # Actor: self.actor + self.actor_optimizer → TrainState
        actor_net = self._make_actor(state_dim, action_dim)
        dummy_obs = jnp.ones((1, state_dim))
        self.actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_key, dummy_obs),
            tx=optax.adam(lr),
        )

        # Temperature: self.log_alpha + self.alpha_optimizer → TrainState
        alpha_net = Temperature()
        self.alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_key),
            tx=optax.adam(lr),
        )

        # Critics: self.q1/q2 + targets + optimizers → CriticTrainState
        critic_net = self._make_critic(state_dim)
        dummy_act = jnp.ones((1, action_dim))
        critic_params = critic_net.init(critic_key, dummy_obs, dummy_act)
        self.critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params,
            target_params=critic_params,
            tx=optax.adam(lr),
        )

    def _make_actor(self, state_dim, action_dim):
        return ActorNetwork(self.hidden_dim, self.depth, action_dim)

    def _make_critic(self, state_dim):
        return JAXConcatQNetwork(self.hidden_dim, self.depth)

    def sample_action(self, obs):
        self.key, action_key = jax.random.split(self.key)
        actions, _ = _select_actions(self.actor, obs, action_key)
        return np.asarray(actions)

    def update(self, states, actions, rewards, next_states, dones):
        # Convert from torch/numpy to JAX arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)

        # Call JIT-compiled pure function
        self.actor, self.critic, self.alpha, self.key, logs = _update_pure(
            self.actor, self.critic, self.alpha, self.key,
            states, actions, rewards, next_states, dones,
            self.num_tasks, self.use_task_vmap, self.gamma, self.tau, self.target_entropy,
        )

        self.training_step += 1

        return {k: float(v) for k, v in logs.items()}
