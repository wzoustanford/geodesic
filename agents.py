from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from typing import Dict, Tuple
from models import BinaryActionQNetwork, MultinomialActionQNetwork
from dataset import Dataset

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
#   _make_network(state_dim) -> nn.Module
#   _transform_actions(actions) -> Tensor (dataset actions → network format)
#   select_actions(states) -> Tensor
#   compute_cql_loss(states, actions, q_network) -> Tensor
#
# DiscreteQLAgent adds discrete-specific hooks and provides
# select_actions / compute_cql_loss via exhaustive enumeration:
#   _num_actions() -> int
#   _all_action_tensors(batch_size) -> Tensor
#   _indices_to_actions(indices) -> Tensor
#
# Future: ContinuousQLAgent would override select_actions with sampling
#
# ============================================================================

class BaseQLAgent(Agent):
    """
    Shared base for Q-learning agents with double Q-networks.
    Handles: __init__, update, _update_single_q, _soft_update_targets, save, load, train.
    Subclasses must implement: _make_network, _transform_actions, select_actions, compute_cql_loss.
    """

    def __init__(
        self,
        state_dim: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.8,
        lr: float = 3e-3,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device

        # Build Q-networks via subclass factory
        self.q1 = self._make_network(state_dim).to(device)
        self.q2 = self._make_network(state_dim).to(device)
        self.q1_target = self._make_network(state_dim).to(device)
        self.q2_target = self._make_network(state_dim).to(device)

        # Initialize targets to match online networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr, weight_decay=1e-5)

        self.training_step = 0

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_network(self, state_dim: int) -> nn.Module:
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

    @abstractmethod
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_network: nn.Module
    ) -> torch.Tensor:
        """
        Compute CQL conservative penalty for a single Q-network.
        CQL penalty = E_s[ logsumexp_a Q(s,a) ] - E_s[ Q(s, a_data) ]

        Discrete agents: exact logsumexp over all actions
        Continuous agents: sampled approximation
        """
        ...

    # ------------------------------------------------------------------
    # Shared: update()
    # ------------------------------------------------------------------
    #
    # Note on target networks: Binary originally used online networks
    # to select next actions (Double DQN), while Dual used target networks
    # for both selection and evaluation. We follow the Double DQN approach
    # (select with online via select_actions, evaluate with target nets)
    # as it's more standard and reduces overestimation bias.
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
        Update Q-networks using TD learning + optional CQL penalty.

        Args:
            states: [batch_size, state_dim]
            actions: raw actions from dataset (converted via _transform_actions)
            rewards: [batch_size]
            next_states: [batch_size, state_dim]
            dones: [batch_size] terminal flags (0 or 1)

        Returns:
            Dictionary of loss metrics
        """
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
    # Shared: save() / load()
    # ------------------------------------------------------------------

    def _extra_save_state(self) -> dict:
        """Override to add subclass-specific fields to checkpoint."""
        return {}
    
    def _load_extra_state(self, checkpoint: dict):
        """Override to restore subclass-specific fields from checkpoint."""
        pass

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

    ## [Note: CQL should be for completeness only, we don't use the loss in experiments]
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

    def _make_network(self, state_dim: int) -> nn.Module:
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

    def _make_network(self, state_dim: int) -> nn.Module:
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
            torch.bucketize(actions[:, 1], bin_edges, right=True),
            0, self.a2_bins - 1
        )
        return a1_idx * self.a2_bins + a2_bin

    def _extra_save_state(self) -> dict:
        return {'a2_bins': self.a2_bins}

    def _load_extra_state(self, checkpoint: dict):
        self.a2_bins = checkpoint.get('a2_bins', self.a2_bins)
    
    # MultinomialAction-specific helpers (not part of the base interface)

    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        """Convert [a1, a2] → discrete action index."""
        a1, a2 = continuous_action
        a1_idx = int(a1)
        a2_bin = np.clip(np.digitize(a2, self.a2_bin_edges) - 1,
                          0, self.a2_bins - 1)
        return a1_idx * self.a2_bins + a2_bin

    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index → [a1, a2]."""
        a1_idx = action_idx // self.a2_bins
        a2_bin = action_idx % self.a2_bins
        a2_value = (self.a2_bin_edges[a2_bin] + self.a2_bin_edges[a2_bin + 1]) / 2
        return np.array([float(a1_idx), a2_value])
