from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from typing import Dict, Tuple
from models import BinaryActionQNetwork, MultinomialActionQNetwork, ConcatQNetwork


# ============================================================================
# Abstract Base Agent
# ============================================================================

class Agent(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def select_actions(self):
        pass
    @abstractmethod
    def update(self):
        pass
    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def save(self):
        pass
    @abstractmethod
    def load(self):
        pass

# ============================================================================
# Base Q-Learning Agent
# ============================================================================
#
# BaseQLAgent abstract hooks (all subclasses):
#   _make_critic(state_dim) -> nn.Module
#   _transform_actions(actions) -> Tensor (dataset actions → network format)
#   select_actions(states) -> Tensor
#
# ============================================================================

class BaseQLAgent(Agent):
    # [extension todo: have abstraction on the critic model architecture from BaseQLAgent]
    """
    Shared base for Q-learning agents with double Q-networks.
    Handles: __init__, update, _update_single_q, _soft_update_targets, save, load, train.
    Subclasses must implement: _make_critic, _transform_actions, select_actions, compute_cql_loss.
    """

    def __init__(
        self,
        state_dim: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.8,
        lr: float = 3e-3,
        grad_clip: float = 1.0,
        save_dir='./checkpoints/',
        experiment_prefix='rl_',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device

        # Build Q-networks via subclass factory
        self.q1 = self._make_critic(state_dim).to(device)
        self.q2 = self._make_critic(state_dim).to(device)
        self.q1_target = self._make_critic(state_dim).to(device)
        self.q2_target = self._make_critic(state_dim).to(device)

        # Initialize targets to match online networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr, weight_decay=1e-5)

        # Training loop variables 
        self.training_step = 0
        self.best_val_q = -float('inf')
        self.save_dir = save_dir
        if self.save_dir == './checkpoints/':
            os.system('mkdir ./checkpoints/')
        self.experiment_prefix = experiment_prefix


    # ------------------------------------------------------------------
    # Abstract hooks — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_critic(self, state_dim: int) -> nn.Module:
        """Return a new Q-network instance (not yet moved to device)."""
        ...

    @abstractmethod
    def _transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert dataset-format actions to Q-network-format actions.
        Called at the start of update() to translate whatever the dataset
        provides into the format the Q-network's forward() expects.

        For discrete agents: dataset actions → discrete indices
        For continuous agents: identity or normalization
        """
        ...

    @abstractmethod
    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Select best actions for a batch of states.

        Discrete agents: exhaustive enumeration over all actions
        Continuous agents: sampling-based (e.g. random shooting, CEM)
        """
        ...

    # ------------------------------------------------------------------
    # Shared: update()
    # ------------------------------------------------------------------
    #
    # Note on target networks: We follow the Double DQN approach
    # (select with online via select_actions with q nets, evaluate with target nets).
    #

    def _update_single_q(
        self,
        q_net: nn.Module,
        optimizer: optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one update step on a single Q-network.

        Returns:
            (td_loss, cql_loss, total_loss) — all as tensors for metric logging
        """
        current_q = q_net(states, actions).squeeze()
        td_loss = F.mse_loss(current_q, target_q)

        if self.alpha > 0:
            cql_loss = self.compute_cql_loss(states, actions, q_net)
            total_loss = td_loss + self.alpha * cql_loss
        else:
            cql_loss = torch.tensor(0.0)
            total_loss = td_loss

        optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), self.grad_clip)
        optimizer.step()

        return td_loss, cql_loss, total_loss

    def _soft_update_targets(self):
        """Polyak-average online networks into target networks."""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """ 
        Update Q-networks using TD learning 

        Args: 
            states: [batch_size, state_dim] 
            actions: raw actions from dataset (converted via _transform_actions) 
            rewards: [batch_size] 
            next_states: [batch_size, state_dim] 
            dones: [batch_size] terminal flags (0 or 1) 

        Returns:
            Dictionary of loss metrics
        """

        # set train status for sub models 
        self.q1.train()
        self.q2.train()
        self.q1_target.train()
        self.q2_target.train()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1. Convert dataset actions to network-ready format
        actions = self._transform_actions(actions)

        # 2. Compute target Q-values (Double DQN: select with online, evaluate with target)
        with torch.no_grad():
            next_actions = self.select_actions(next_states)
            next_q1 = self.q1_target(next_states, next_actions).squeeze()
            next_q2 = self.q2_target(next_states, next_actions).squeeze()
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 3. Update both Q-networks
        q1_loss, cql1_loss, total_q1_loss = self._update_single_q(
            self.q1, self.q1_optimizer, states, actions, target_q
        )
        q2_loss, cql2_loss, total_q2_loss = self._update_single_q(
            self.q2, self.q2_optimizer, states, actions, target_q
        )

        # 4. Soft-update target networks
        self._soft_update_targets()

        self.training_step += 1

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if self.alpha > 0 else 0.0,
            'cql2_loss': cql2_loss.item() if self.alpha > 0 else 0.0,
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item(),
        }

    # ------------------------------------------------------------------
    # Shared: validate() 
    # ------------------------------------------------------------------
    def validate(self, states, actions):
        # Validation phase
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            
            # Convert continuous actions to discrete for validation 
            actions = self._transform_actions(actions)

            val_q1 = self.q1(states, actions).squeeze()
            val_q2 = self.q2(states, actions).squeeze()
            val_q = torch.min(val_q1, val_q2)
            
            val_q = val_q.mean().item()
        
        self.save_best_val_q_model(val_q, self.get_save_path('best'))
        
        return val_q

    # ------------------------------------------------------------------
    # Shared: save() / load()
    # ------------------------------------------------------------------

    def _extra_save_state(self) -> dict:
        """Override to add subclass-specific fields to checkpoint."""
        return {}

    def _load_extra_state(self, checkpoint: dict):
        """Override to restore subclass-specific fields from checkpoint."""
        pass

    def get_save_path(self, postfix: str):
        return f'{self.save_dir}/{self.experiment_prefix}_{postfix}.pt'

    def save_best_val_q_model(self, val_q: float, path: str):
        # Save best model
        if val_q > self.best_val_q:
            print('<'*3+f'best q val model found. val_loss: {val_q:.3f}. prev_best_val_loss: {self.best_val_q:.3f}. saving to {path}')
            self.best_val_q = val_q
            self.save(path)

    def save(self, path: str):
        checkpoint = {
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau,
            'training_step': self.training_step,
        }
        checkpoint.update(self._extra_save_state())
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self._load_extra_state(checkpoint)
        print(f"Model loaded from {path}")

    def train(self):
        raise NotImplementedError("train() not yet implemented in BaseQLAgent")

# ============================================================================
# Discrete Q-Learning Agent (intermediate base for finite action spaces)
# ============================================================================
#
# DiscreteQLAgent adds discrete-specific hooks and provides
# select_actions by evaluating Q for all actions: 
#   _num_actions() -> int
#   _all_action_tensors(batch_size) -> Tensor
#   _indices_to_actions(indices) -> Tensor
#
# Future: ContinuousQLAgent would override select_actions with sampling
#
# ============================================================================

class DiscreteQLAgent(BaseQLAgent):
    """
    Intermediate base for Q-learning agents with finite discrete action spaces.
    Provides select_actions (exhaustive enumeration) and compute_cql_loss
    (exact logsumexp) using 3 hooks that concrete subclasses implement.
    """

    # ------------------------------------------------------------------
    # Abstract hooks — discrete subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _num_actions(self) -> int:
        """Total number of discrete actions to enumerate."""
        ...

    @abstractmethod
    def _all_action_tensors(self, batch_size: int) -> torch.Tensor:
        """
        Return action tensors for ALL possible actions, repeated for each
        element in the batch, formatted for the Q-network's forward().

        Shape depends on network type:
          - Binary: [batch_size * 2, 1]  (float actions)
          - Dual:   [batch_size * N]     (integer indices)
        """
        ...

    # ------------------------------------------------------------------
    # Discrete: select_actions (exhaustive enumeration)
    # ------------------------------------------------------------------

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Select best actions for a batch of states via exhaustive Q-evaluation.
        Works for any finite discrete action space.
        """
        with torch.no_grad():
            batch_size = states.shape[0]
            n = self._num_actions()

            all_actions = self._all_action_tensors(batch_size)
            states_exp = states.unsqueeze(1).expand(-1, n, -1).reshape(-1, self.state_dim)

            q1_vals = self.q1(states_exp, all_actions).reshape(batch_size, n)
            q2_vals = self.q2(states_exp, all_actions).reshape(batch_size, n)
            q_vals = torch.min(q1_vals, q2_vals)

            return q_vals.argmax(dim=1) # returns the best action indices
    
    # ------------------------------------------------------------------
    # Discrete: compute_cql_loss (exact logsumexp over all actions)
    # ------------------------------------------------------------------

    ## [Note: CQL should be for completeness only, we may not use the loss in experiments]
    ## [Note: there is no temperature definition in the cql code] 
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_network: nn.Module
    ) -> torch.Tensor:
        batch_size = states.shape[0]
        n = self._num_actions()

        current_q = q_network(states, actions).squeeze()

        all_actions = self._all_action_tensors(batch_size)
        states_exp = states.unsqueeze(1).expand(-1, n, -1).reshape(-1, self.state_dim)
        all_q = q_network(states_exp, all_actions).reshape(batch_size, n)

        logsumexp = torch.logsumexp(all_q, dim=1)
        cql_loss = (logsumexp - current_q).mean()
        return cql_loss

# ============================================================================
# Binary Action QL Agent
# ============================================================================

class BinaryActionQLAgent(DiscreteQLAgent):
    """Q-Learning agent for binary action spaces (action = 0 or 1)."""

    def __init__(self, state_dim: int, **kwargs):
        super().__init__(state_dim, **kwargs)

    def _make_critic(self, state_dim: int) -> nn.Module:
        return BinaryActionQNetwork(state_dim)

    def _num_actions(self) -> int:
        return 2

    def _all_action_tensors(self, batch_size: int) -> torch.Tensor:
        # Two possible actions: 0.0 and 1.0
        # BinaryActionQNetwork.forward(state, action) expects action as [batch, 1] float
        # Return shape: [batch_size * 2, 1]
        actions_0 = torch.zeros(batch_size, 1, device=self.device)
        actions_1 = torch.ones(batch_size, 1, device=self.device)
        return torch.cat([actions_0, actions_1], dim=0)
    
    def _transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # Dataset gives float 0.0/1.0, network expects [batch_size, 1]
        return actions.unsqueeze(1) if actions.dim() == 1 else actions

# ============================================================================
# Discrete Action QL Agent
# ============================================================================

class MultinomialActionQLAgent(DiscreteQLAgent):
    """
    Q-Learning agent for discrete action space
    A1: Binary (0 or 1)
    A2: Discretized into bins
    Total actions: 2 * a2_bins
    """

    def __init__(self, state_dim: int, a2_bins: int = 5, **kwargs):
        self.a2_bins = a2_bins
        self.total_actions = 2 * a2_bins
        self.a2_bin_edges = np.linspace(0, 0.5, a2_bins + 1)
        super().__init__(state_dim, **kwargs)

    def _make_critic(self, state_dim: int) -> nn.Module:
        return MultinomialActionQNetwork(state_dim, self.a2_bins)

    def _num_actions(self) -> int:
        return self.total_actions

    def _all_action_tensors(self, batch_size: int) -> torch.Tensor:
        # MultinomialActionQNetwork.forward(state, action_idx) expects
        # action_idx as [batch] long tensor
        # Return shape: [batch_size * total_actions]
        single = torch.arange(self.total_actions, device=self.device)
        return single.unsqueeze(0).expand(batch_size, -1).reshape(-1)

    def _transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # Dataset gives continuous [a1, a2] float pairs, shape [batch_size, 2]
        # Network expects discrete long indices, shape [batch_size]
        # Vectorized replacement for the original per-element for-loop
        a1_idx = actions[:, 0].long()
        bin_edges = torch.tensor(self.a2_bin_edges[1:], device=actions.device)
        # right=True matches np.digitize behavior: exact boundary → higher bin
        a2_bin = torch.clamp(
            torch.bucketize(actions[:, 1].contiguous(), bin_edges, right=True),
            0, self.a2_bins - 1
        )
        return a1_idx * self.a2_bins + a2_bin

    def _extra_save_state(self) -> dict:
        return {'a2_bins': self.a2_bins}

    def _load_extra_state(self, checkpoint: dict):
        self.a2_bins = checkpoint.get('a2_bins', self.a2_bins)
    
    def get_save_path(self, postfix: str):
        return f'{self.save_dir}/{self.experiment_prefix}_a2_bins{self.a2_bins}_{postfix}.pt'

    # MultinomialAction-specific helpers 

    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        # [note] helper function, not used 
        """Convert [a1, a2] → discrete action index."""
        a1, a2 = continuous_action
        a1_idx = int(a1)
        a2_bin = np.clip(np.digitize(a2, self.a2_bin_edges) - 1,
                          0, self.a2_bins - 1)
        return a1_idx * self.a2_bins + a2_bin

    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        # [note] helper function, not used
        """Convert discrete action index → [a1, a2]."""
        a1_idx = action_idx // self.a2_bins
        a2_bin = action_idx % self.a2_bins
        a2_value = (self.a2_bin_edges[a2_bin] + self.a2_bin_edges[a2_bin + 1]) / 2
        return np.array([float(a1_idx), a2_value])

# ============================================================================
# SAC Agent (Soft Actor-Critic for continuous action spaces)
# ============================================================================

class SACAgent(BaseQLAgent):
    """
    Soft Actor-Critic agent for continuous action spaces.
    Adds actor (policy) network and learnable temperature on top of BaseQLAgent.
    Overrides update() with actor-critic-alpha three-step optimization.

    Handling task IDs:
    - Task IDs are NOT stored as a separate field. They are embedded in the
      observation as a one-hot vector (last num_tasks dims) by the vectorized
      env when use_one_hot=True.
    - In start_online(), the env returns obs of shape (num_envs, obs_dim) where
      each env's obs already contains its one-hot task encoding. This flows
      through add_transition -> _store_window -> DataLoader -> update()
      automatically.
    - When batches are flattened from (batch, 1, num_tasks, obs_dim) to
      (N, obs_dim), the one-hot task encoding is preserved in each row.
    - To extract task IDs at any point (e.g. for per-task evaluation or
      mixture-of-experts routing like Moore): task_ids = obs[..., -num_tasks:]
    - This matches the reference (metaworld-algorithms mtsac.py:455):
      task_ids = data.observations[..., -self.num_tasks:]
    """

    def __init__(self, state_dim, action_dim, hidden_dim=400, depth=3, **kwargs):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        super().__init__(state_dim, **kwargs)

        self.actor = self._make_actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=kwargs.get('lr', 3e-3))

        # Learnable temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=kwargs.get('lr', 3e-3))
        self.target_entropy = -action_dim

        # Critic list for looping
        self.critics = [self.q1, self.q2]
        self.critic_targets = [self.q1_target, self.q2_target]
        self.critic_optimizers = [self.q1_optimizer, self.q2_optimizer]

    def _make_actor(self, state_dim, action_dim) -> nn.Module:
        """Return a policy network. Override for custom architectures."""
        layers = []
        in_dim = state_dim
        for _ in range(self.depth):
            layers += [nn.Linear(in_dim, self.hidden_dim), nn.ReLU()]
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, action_dim * 2))
        return nn.Sequential(*layers)

    def _make_critic(self, state_dim) -> nn.Module:
        """Q-network for continuous actions: takes (state, action) as two args."""
        return ConcatQNetwork(state_dim, self.action_dim, self.hidden_dim, self.depth)

    def _transform_actions(self, actions):
        return actions

    def _get_action_dist(self, states):
        output = self.actor(states)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        return mean, log_std.exp()

    def select_actions(self, states):
        """Sample actions from policy. Returns (actions, log_probs)."""
        mean, std = self._get_action_dist(states)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        actions = torch.tanh(raw)
        log_probs = dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        return actions, log_probs

    def sample_action(self, states):
        """For env interaction — no grad, tensor out."""
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions, _ = self.select_actions(states)
        return actions

    # ------------------------------------------------------------------
    # Critic helpers
    # ------------------------------------------------------------------

    def compute_target_q(self, next_states, next_actions, rewards, dones, alpha, next_log_probs):
        with torch.no_grad():
            target_qs = [t(next_states, next_actions) for t in self.critic_targets]
            min_q = torch.min(torch.stack(target_qs), dim=0).values.squeeze(-1)
            next_log_probs = next_log_probs.squeeze(-1)
            return rewards + self.gamma * (1 - dones) * (min_q - alpha * next_log_probs)

    def critic_loss(self, states, actions, target_q):
        return [F.mse_loss(q(states, actions).squeeze(-1), target_q) for q in self.critics]

    def critic_step(self, losses):
        for opt, loss in zip(self.critic_optimizers, losses):
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ------------------------------------------------------------------
    # update() — overrides BaseQLAgent
    # ------------------------------------------------------------------

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        sac_alpha = self.log_alpha.exp().detach()

        # --- 1. Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.select_actions(next_states)
        target_q = self.compute_target_q(next_states, next_actions, rewards, dones, sac_alpha, next_log_probs)
        losses = self.critic_loss(states, actions, target_q)
        self.critic_step(losses)

        # --- 2. Actor update ---
        actions_pred, log_probs = self.select_actions(states)
        q_vals = [q(states, actions_pred) for q in self.critics]
        min_q = torch.min(torch.stack(q_vals), dim=0).values
        actor_loss = (sac_alpha * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        self.actor_optimizer.step()

        # --- 3. Alpha (temperature) update ---
        alpha_loss = (-self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 4. Soft update targets ---
        self._soft_update_targets()

        self.training_step += 1

        # --- Logging ---
        actor_param_norm = torch.cat([p.view(-1) for p in self.actor.parameters()]).norm()
        critic_param_norm = torch.cat([
            p.view(-1) for q in self.critics for p in q.parameters()
        ]).norm()

        return {
            'q1_loss': losses[0].item(),
            'q2_loss': losses[1].item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'sac_alpha': sac_alpha.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'actor_param_norm': actor_param_norm.item(),
            'critic_param_norm': critic_param_norm.item(),
        }
