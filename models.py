import torch
from torch import nn

import torch.nn.functional as F
import ray

# ============================================================================
# Binary Action QL Model
# ============================================================================

class BinaryActionQNetwork(nn.Module):
    """Q-network for state, actions, Q(s,a)"""
    """dnn architecture"""
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = state_dim + 1 # add 1 dim for binary action 
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = torch.reshape(action, (-1, 1))
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# ============================================================================
# Discrete Action QL Model
# ============================================================================

class MultinomialActionQNetwork(nn.Module):
    """
    Q-network for block discrete actions, with Q(s,a) 
    Takes state and discrete action index as input, outputs single Q-value
    A1: 2 actions (binary)
    A2: a2_bins actions (discretized continuous)
    Total: 2 * a2_bins possible actions
    """
    
    def __init__(self, state_dim: int, a2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.a2_bins = a2_bins
        self.total_actions = 2 * a2_bins  # a1 (2) x a2 (bins)
        
        # Network takes state + one-hot encoded action
        input_dim = state_dim + self.total_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)  # Output single Q-value
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action_idx: [batch_size] - discrete action indices (0 to total_actions-1)
        Returns:
            q_value: [batch_size, 1] - Q-value for each (state, action) pair
        """
        
        # Convert action indices to one-hot encoding
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        
        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=-1)
        
        # Forward through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# Continuous Action Q Model (for SAC)
# ============================================================================

class ConcatQNetwork(nn.Module):
    """Q-network that concatenates state and action before the MLP."""
    def __init__(self, state_dim, action_dim, hidden_dim, depth):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, states, actions):
        return self.mlp(torch.cat([states, actions], dim=-1))


# ============================================================================
# Shared Storage for Parallel Training
# ============================================================================

@ray.remote
class ModelSharedStorage:
    """Ray actor that stores model weights for worker synchronization.

    The training loop pushes updated actor params after gradient steps.
    DataWorkers pull actor params to run the policy in their environments.
    As a Ray actor, all method calls are serialized — no explicit locks needed.

    For JAX agents, only params (pytrees of arrays) are stored — not apply_fn
    or opt_state, which are non-serializable and only needed by the trainer.
    Workers reconstruct a local TrainState shell and swap in the params.
    """
    def __init__(self):
        self.weights = None
        self.step_counter = 0
        self.warmstart_done = False

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_counter(self):
        return self.step_counter

    def incr_counter(self):
        self.step_counter += 1

    def get_warmstart_signal(self):
        return self.warmstart_done

    def set_warmstart_signal(self):
        self.warmstart_done = True
