from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from typing import Dict, Tuple
from models import BinaryActionQNetwork, DualDiscreteActionQNetwork
from dataset import Dataset

class Agent(ABC):
    def __init__(self):
        pass 
    @abstractmethod
    def select_action(self):
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

class BinaryActionQLAgent(Agent):
    """
    Q-Learning for binary actions 
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
        """
        Initialize Binary QL 
        
        Args:
            state_dim: Dimension of state space
            alpha: conservative penalty weight
            gamma: Discount factor
            tau: Target network update rate
            lr: Learning rate
            grad_clip: Gradient clipping value
            device: Device 
        """
        self.state_dim = state_dim
        self.action_dim = 1  # Binary action
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks 
        self.q1 = BinaryActionQNetwork(state_dim, self.action_dim).to(device)
        self.q2 = BinaryActionQNetwork(state_dim, self.action_dim).to(device)
        self.q1_target = BinaryActionQNetwork(state_dim, self.action_dim).to(device)
        self.q2_target = BinaryActionQNetwork(state_dim, self.action_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr, weight_decay=1e-5)
        
        # Track training metrics
        self.training_step = 0
    
    def select_actions_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Select best actions for a batch of states efficiently.
        Evaluates Q(s,0) and Q(s,1) for each state and returns argmax.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            Best actions [batch_size, 1]
        """

        # check for whether use epsilon for exploration 

        batch_size = states.shape[0]
        
        with torch.no_grad():
            # Create action tensors for both possible actions
            actions_0 = torch.zeros(batch_size, 1).to(self.device)
            actions_1 = torch.ones(batch_size, 1).to(self.device)
            
            # Evaluate Q-values for action=0
            q1_values_0 = self.q1(states, actions_0).squeeze()
            q2_values_0 = self.q2(states, actions_0).squeeze()
            q_values_0 = torch.min(q1_values_0, q2_values_0)
            
            # Evaluate Q-values for action=1
            q1_values_1 = self.q1(states, actions_1).squeeze()
            q2_values_1 = self.q2(states, actions_1).squeeze()
            q_values_1 = torch.min(q1_values_1, q2_values_1)
            
            # Select best action for each state (argmax)
            best_actions = (q_values_1 > q_values_0).float().unsqueeze(1)
            
            return best_actions
    
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_network: nn.Module
    ) -> torch.Tensor:
        """
        Compute CQL penalty for binary actions.
        Since we only have 2 actions, we compute logsumexp over both.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            q_network: Q-network to compute loss for
            
        Returns:
            CQL loss value
        """
        batch_size = states.shape[0]
        
        # Current Q-values for taken actions
        current_q = q_network(states, actions).squeeze()
        
        # Q-values for both possible actions
        with torch.no_grad():
            actions_0 = torch.zeros(batch_size, 1).to(self.device)
            actions_1 = torch.ones(batch_size, 1).to(self.device)
        
        q_0 = q_network(states, actions_0).squeeze()
        q_1 = q_network(states, actions_1).squeeze()
        
        # Stack Q-values for both actions
        all_q = torch.stack([q_0, q_1], dim=1)  # [batch_size, 2]
        
        # Conservative penalty: log-sum-exp of all actions minus current Q 
        # Temperature scaling for numerical stability 
        logsumexp = torch.logsumexp(all_q / 10.0, dim=1) * 10.0 
        # [todo] double check whether this loss is still used, if so check that the temperature values are aigned across multiple algorithms 
        cql_loss = (logsumexp - current_q).mean()
        
        return cql_loss
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update Q-networks with QL
        
        Args:
            states: Current states
            actions: Actions taken (should be 0 or 1)
            rewards: Rewards received
            next_states: Next states
            dones: Done flags
            
        Returns:
            Dictionary of losses
        """
        # Ensure actions are properly shaped [batch_size, 1]
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        
        #Clip rewards for stability
        #rewards = torch.clamp(rewards, -10, 10)
        
        # Compute target Q-values
        with torch.no_grad():
            # Select best next actions using batch selection
            next_actions = self.select_actions_batch(next_states)
            
            # Get target Q-values (using target networks for stability)
            next_q1 = self.q1_target(next_states, next_actions).squeeze()
            next_q2 = self.q2_target(next_states, next_actions).squeeze()
            next_q = torch.min(next_q1, next_q2)
            
            # Compute targets
            target_q = rewards + self.gamma * next_q * (1 - dones)
            #target_q = torch.clamp(target_q, -50, 50)
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        if self.alpha > 0:
            cql1_loss = self.compute_cql_loss(states, actions, self.q1)
            total_q1_loss = q1_loss + self.alpha * cql1_loss
        else:
            cql1_loss = torch.tensor(0.0)
            total_q1_loss = q1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        if self.alpha > 0:
            cql2_loss = self.compute_cql_loss(states, actions, self.q2)
            total_q2_loss = q2_loss + self.alpha * cql2_loss
        else:
            cql2_loss = torch.tensor(0.0)
            total_q2_loss = q2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Update target networks with soft update
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_step += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if self.alpha > 0 else 0.0,
            'cql2_loss': cql2_loss.item() if self.alpha > 0 else 0.0,
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item()
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau,
            'training_step': self.training_step
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {path}")

    def train(
            self,
            dataset: Dataset, 
            alpha: float = 1.0,
            epochs: int = 100,
            batch_size: int = 256,
            lr: float = 1e-3,
            output_dir: str = 'experiment',
            model_name: str = None
        ) -> None:
        """
        Train Binary QL with continuous Q-network architecture
        
        Args:
            alpha: CQL penalty weight
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            output_dir: Directory to save models
            model_name: Name for saved model (auto-generated if None)
            
        Returns:
            Trained agent and data pipeline
        """
        print("="*70)
        print(" BINARY CQL TRAINING WITH CONTINUOUS Q-NETWORK")
        print("="*70)
        print(f"\nHyperparameters:")
        print(f"  Alpha (CQL penalty): {alpha}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print("="*70)
        
        # Initialize data pipeline
        print("\nInitializing data pipeline...")
        train_data, val_data, test_data = dataset.prepare_data()
        
        # Get state dimension
        state_dim = train_data['states'].shape[1]
        print(f"\nState dimension: {state_dim}")
        print(f"Action dimension: 1 (binary: 0 or 1)")
        
        # Training loop
        print(f"\n{'='*70}")
        print(" TRAINING")
        print(f"{'='*70}")
        
        best_val_loss = float('inf')
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name is None:
            model_name = f"binary_ql"
        
        for epoch in range(epochs):
            # Training phase
            self.q1.train()
            self.q2.train()
            
            train_metrics = {
                'q1_loss': 0, 'q2_loss': 0,
                'cql1_loss': 0, 'cql2_loss': 0,
                'total_q1_loss': 0, 'total_q2_loss': 0
            }
            
            # Sample random batches for training
            n_batches = len(train_data['states']) // batch_size
            
            for _ in range(n_batches):
                # Get batch
                batch = dataset.get_batch(batch_size=batch_size, split='train')
                
                # Convert to tensors
                states = torch.FloatTensor(batch['states']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                rewards = torch.FloatTensor(batch['rewards']).to(self.device)
                next_states = torch.FloatTensor(batch['next_states']).to(self.device)
                dones = torch.FloatTensor(batch['dones']).to(self.device)
                
                # Update self
                metrics = self.update(states, actions, rewards, next_states, dones)
                
                # Accumulate metrics
                for key in train_metrics:
                    train_metrics[key] += metrics.get(key, 0)
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= n_batches
            
            # Validation phase
            self.q1.eval()
            self.q2.eval()
            
            val_q_values = []
            with torch.no_grad():
                # Sample validation batches
                for _ in range(10):  # Use 10 validation batches
                    batch = dataset.get_batch(batch_size=batch_size, split='val')
                    
                    states = torch.FloatTensor(batch['states']).to(self.device)
                    actions = torch.FloatTensor(batch['actions']).to(self.device)
                    
                    if actions.dim() == 1:
                        actions = actions.unsqueeze(1)
                    
                    q1_val = self.q1(states, actions).squeeze()
                    q2_val = self.q2(states, actions).squeeze()
                    q_val = torch.min(q1_val, q2_val)
                    
                    val_q_values.append(q_val.mean().item())
            
            val_loss = -np.mean(val_q_values)  # Negative because we want higher Q-values
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(os.path.join(output_dir, f'{model_name}_best.pt'))
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train - Q1 Loss: {train_metrics['q1_loss']:.4f}, "
                    f"CQL1 Loss: {train_metrics['cql1_loss']:.4f}, "
                    f"Total Q1: {train_metrics['total_q1_loss']:.4f}")
                print(f"  Val   - Avg Q-value: {-val_loss:.4f}")
        
        # Save final model
        self.save(os.path.join(output_dir, f'{model_name}_final.pt'))
        
        print(f"\n{'='*70}")
        print(" TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best model saved to: {output_dir}/{model_name}_best.pt")
        print(f"Final model saved to: {output_dir}/{model_name}_final.pt")
        
        return

class DualDiscreteActionQLAgent:
    """
    Block Discrete CQL agent with discretized VP2 action space
    VP1: Binary (0 or 1) 
    VP2: Discretized into bins (0 to 0.5 mcg/kg/min)
    """
    
    def __init__(self, state_dim: int, vp2_bins: int = 5, alpha: float = 1.0, 
                 gamma: float = 0.95, tau: float = 0.8, lr: float = 1e-3, 
                 grad_clip: float = 1.0):
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins  # VP1 (2) x VP2 (bins)
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        
        # Define VP2 bin edges (0 to 0.5)
        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-networks
        self.q1 = DualDiscreteActionQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2 = DualDiscreteActionQNetwork(state_dim, vp2_bins).to(self.device)
        
        # Initialize target networks
        self.q1_target = DualDiscreteActionQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2_target = DualDiscreteActionQNetwork(state_dim, vp2_bins).to(self.device)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Initialize optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action [vp1, vp2] to discrete action index
        Args:
            continuous_action: [vp1, vp2] where vp1 is binary, vp2 is continuous
        Returns:
            action_idx: discrete action index (0 to total_actions-1)
        """
        vp1, vp2 = continuous_action
        vp1_idx = int(vp1)  # 0 or 1
        
        # Find which bin vp2 falls into
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        
        # Combine into single action index
        action_idx = vp1_idx * self.vp2_bins + vp2_bin
        return action_idx
    
    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        """
        Convert discrete action index to continuous action [vp1, vp2]
        Args:
            action_idx: discrete action index (0 to total_actions-1)
        Returns:
            continuous_action: [vp1, vp2] where vp1 is binary, vp2 is continuous
        """
        vp1_idx = action_idx // self.vp2_bins
        vp2_bin = action_idx % self.vp2_bins
        
        # Convert bin to continuous value (use bin center)
        vp2_value = (self.vp2_bin_edges[vp2_bin] + self.vp2_bin_edges[vp2_bin + 1]) / 2
        
        return np.array([float(vp1_idx), vp2_value])
    
    def select_action(self, state: np.ndarray, num_samples: int = 50) -> np.ndarray:
        """
        Select best action using Q-values over all discrete actions
        Optimized with batch processing - no for-loops
        """
        with torch.no_grad():
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            batch_size = state.shape[0]
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Create all possible discrete actions for each state in batch
            # Shape: [batch_size, total_actions]
            all_actions = torch.arange(self.total_actions).to(self.device)
            all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)
            
            # Expand states to match actions
            # Shape: [batch_size * total_actions, state_dim]
            state_expanded = state_tensor.unsqueeze(1).expand(-1, self.total_actions, -1)
            state_expanded = state_expanded.reshape(-1, self.state_dim)
            
            # Flatten actions for network input
            # Shape: [batch_size * total_actions]
            actions_flat = all_actions.reshape(-1)
            
            # Compute Q-values for all actions
            q1_values = self.q1(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
            q2_values = self.q2(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
            q_values = torch.min(q1_values, q2_values)
            
            # Get best action for each batch element
            best_action_indices = q_values.argmax(dim=1).cpu().numpy()
            
            # Vectorized conversion from discrete indices to continuous actions
            # Extract VP1 (binary) and VP2 bin indices
            vp1_actions = (best_action_indices // self.vp2_bins).astype(float)
            vp2_bin_indices = best_action_indices % self.vp2_bins
            
            # Convert VP2 bin indices to continuous values using bin centers
            vp2_bin_centers = (self.vp2_bin_edges[:-1] + self.vp2_bin_edges[1:]) / 2
            vp2_actions = vp2_bin_centers[vp2_bin_indices]
            
            # Stack VP1 and VP2 into action array
            actions = np.stack([vp1_actions, vp2_actions], axis=1)
            
            return actions if batch_size > 1 else actions[0]
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               next_states: torch.Tensor, dones: torch.Tensor) -> dict:
        """
        Update Q-networks with TD loss and CQL penalty
        """
        batch_size = states.shape[0]
        
        # Convert continuous actions to discrete indices
        action_indices = []
        for i in range(batch_size):
            action_np = actions[i].cpu().numpy()
            action_idx = self.continuous_to_discrete_action(action_np)
            action_indices.append(action_idx)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            # Get next actions using current Q-networks
            all_next_actions = torch.arange(self.total_actions).to(self.device)
            all_next_actions = all_next_actions.unsqueeze(0).expand(batch_size, -1)
            
            next_states_expanded = next_states.unsqueeze(1).expand(-1, self.total_actions, -1)
            next_states_expanded = next_states_expanded.reshape(-1, self.state_dim)
            next_actions_flat = all_next_actions.reshape(-1)
            
            next_q1 = self.q1_target(next_states_expanded, next_actions_flat).reshape(batch_size, self.total_actions)
            next_q2 = self.q2_target(next_states_expanded, next_actions_flat).reshape(batch_size, self.total_actions)
            next_q = torch.min(next_q1, next_q2)
            
            next_best_actions = next_q.argmax(dim=1)
            
            # Compute target values
            next_q1_target = self.q1_target(next_states, next_best_actions).squeeze()
            next_q2_target = self.q2_target(next_states, next_best_actions).squeeze()
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            target_q = rewards + self.gamma * next_q_target * (1 - dones)
        
        # Update Q1
        current_q1 = self.q1(states, action_indices).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Compute logsumexp over all actions for CQL
            all_actions = torch.arange(self.total_actions).to(self.device)
            all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)
            
            states_expanded = states.unsqueeze(1).expand(-1, self.total_actions, -1)
            states_expanded = states_expanded.reshape(-1, self.state_dim)
            actions_flat = all_actions.reshape(-1)
            
            q1_all = self.q1(states_expanded, actions_flat).reshape(batch_size, self.total_actions)
            cql1_loss = torch.logsumexp(q1_all, dim=1).mean() - current_q1.mean()
        else:
            cql1_loss = torch.tensor(0.0).to(self.device)
        
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        current_q2 = self.q2(states, action_indices).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # CQL penalty for Q2
        if self.alpha > 0:
            q2_all = self.q2(states_expanded, actions_flat).reshape(batch_size, self.total_actions)
            cql2_loss = torch.logsumexp(q2_all, dim=1).mean() - current_q2.mean()
        else:
            cql2_loss = torch.tensor(0.0).to(self.device)
        
        total_q2_loss = q2_loss + self.alpha * cql2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if self.alpha > 0 else 0,
            'cql2_loss': cql2_loss.item() if self.alpha > 0 else 0,
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item()
        }
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'state_dim': self.state_dim,
            'vp2_bins': self.vp2_bins,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau
        }, filepath)

    def train(
        alpha: float = 0.001,
        vp2_bins: int = 5,
        epochs: int = 100,
        reward_model_path: str = None,
        suffix: str = "",
        save_dir: str = "experiment/ql",
        reward_combine_lambda: float = None,
        combined_or_train_data_path: str = None,
        eval_data_path: str = None,
        irl_vp2_bins: int = None
    ):
        """Train Block Discrete CQL with specified alpha and optional learned reward

        Args:
            alpha: CQL penalty strength
            vp2_bins: Number of bins for VP2 discretization (for Q-learning action space)
            epochs: Number of training epochs
            reward_model_path: Path to learned reward model (gcl/iq_learn/maxent/unet)
            suffix: Suffix to add to experiment prefix
            save_dir: Directory to save models
            reward_combine_lambda: If None, use pure IRL reward. If in [0, 1], use
                (1 - lambda) * manual_reward + lambda * irl_reward.
            combined_or_train_data_path: Path to training dataset. If eval_data_path is
                also provided, all patients are used for training. Otherwise split into
                train/val/test. If None, uses default config.DATA_PATH.
            eval_data_path: Path to evaluation dataset. If provided, enables dual-dataset
                mode where this dataset is split 50/50 into val/test.
            irl_vp2_bins: Number of VP2 bins used to train the IRL model. If None, uses
                the same value as vp2_bins. This allows loading an IRL model trained with
                different discretization than the Q-learning action space.
        """
        # Determine IRL model vp2_bins (default to vp2_bins if not specified)
        if irl_vp2_bins is None:
            irl_vp2_bins = vp2_bins

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize data pipeline and infer reward type
        print("\nInitializing Block Discrete CQL data pipeline...", flush=True)
        if reward_model_path is None:
            reward_type = "manual"
            pipeline = IntegratedDataPipelineV3(
                model_type='dual', reward_source='manual', random_seed=42,
                combined_or_train_data_path=combined_or_train_data_path,
                eval_data_path=eval_data_path
            )
        elif 'gcl' in reward_model_path:
            reward_type = "gcl"
            pipeline = IntegratedDataPipelineV3(
                model_type='dual', reward_source='learned', random_seed=42,
                reward_combine_lambda=reward_combine_lambda,
                combined_or_train_data_path=combined_or_train_data_path,
                eval_data_path=eval_data_path
            )
            pipeline.load_gcl_reward_model(reward_model_path)
        elif 'iq_learn' in reward_model_path:
            reward_type = "iq_learn"
            pipeline = IntegratedDataPipelineV3(
                model_type='dual', reward_source='learned', random_seed=42,
                reward_combine_lambda=reward_combine_lambda,
                combined_or_train_data_path=combined_or_train_data_path,
                eval_data_path=eval_data_path
            )
            pipeline.load_iq_learn_reward_model(reward_model_path)
        elif 'maxent' in reward_model_path:
            reward_type = "maxent"
            pipeline = IntegratedDataPipelineV3(
                model_type='dual', reward_source='learned', random_seed=42,
                reward_combine_lambda=reward_combine_lambda,
                combined_or_train_data_path=combined_or_train_data_path,
                eval_data_path=eval_data_path
            )
            pipeline.load_maxent_reward_model(reward_model_path)
        elif 'semi_supervised_unet' in reward_model_path:
            # Semi-supervised U-Net provides learned rewards via per-trajectory inference
            reward_type = "semi_supervised_unet"
            pipeline = IntegratedDataPipelineV3(
                model_type='dual', reward_source='learned', random_seed=42,
                reward_combine_lambda=reward_combine_lambda,
                combined_or_train_data_path=combined_or_train_data_path,
                eval_data_path=eval_data_path
            )
            # Load Semi-supervised U-Net model (uses only UNetRewardGenerator, not MortalityDiffuser)
            # Use irl_vp2_bins for loading (IRL model's action space), not vp2_bins (Q-learning action space)
            pipeline.load_semi_supervised_unet_reward_model(reward_model_path, vp1_bins=2, vp2_bins=irl_vp2_bins)
            print(f"  IRL model vp2_bins: {irl_vp2_bins}, Q-learning vp2_bins: {vp2_bins}")
        elif 'unet' in reward_model_path:
            # U-Net provides learned rewards via per-trajectory inference
            reward_type = "unet"
            pipeline = IntegratedDataPipelineV3(
                model_type='dual', reward_source='learned', random_seed=42,
                reward_combine_lambda=reward_combine_lambda,
                combined_or_train_data_path=combined_or_train_data_path,
                eval_data_path=eval_data_path
            )
            # Load U-Net model - use irl_vp2_bins for the IRL model's action space
            # This allows the IRL model to have different discretization than Q-learning
            pipeline.load_unet_reward_model(reward_model_path, vp1_bins=2, vp2_bins=irl_vp2_bins)
            print(f"  IRL model vp2_bins: {irl_vp2_bins}, Q-learning vp2_bins: {vp2_bins}")
        else:
            raise ValueError(f"Cannot infer reward model type from path: {reward_model_path}")

        # Use pipeline's get_reward_prefix for correct naming with lambda
        experiment_prefix = pipeline.get_reward_prefix() if hasattr(pipeline, 'get_reward_prefix') else reward_type
        experiment_prefix = f"{experiment_prefix}{suffix}"

        print("="*70, flush=True)
        print(f" BLOCK DISCRETE CQL TRAINING WITH ALPHA={alpha}", flush=True)
        print(f" Reward: {reward_type} | Prefix: {experiment_prefix}", flush=True)
        print("="*70, flush=True)

        train_data, val_data, test_data = pipeline.prepare_data()
        
        # Get state dimension
        state_dim = train_data['states'].shape[1]

        # Print settings
        print("\n" + "="*70, flush=True)
        print("SETTINGS:", flush=True)
        print(f"  State dimension: {state_dim}", flush=True)
        print(f"  Action dimension: 2 (VP1: binary, VP2: {vp2_bins} bins)", flush=True)
        print(f"  Total discrete actions: {2 * vp2_bins}", flush=True)
        print(f"  ALPHA = {alpha}", flush=True)
        print(f"  TAU = 0.8 (target network update)", flush=True)
        print("  LR = 0.001 (learning rate)", flush=True)
        print("  BATCH_SIZE = 128", flush=True)
        print("  EPOCHS = 100", flush=True)
        print("="*70, flush=True)
        
        # Initialize agent with specified parameters
        agent = DualBlockDiscreteCQL(
            state_dim=state_dim,
            vp2_bins=vp2_bins,
            alpha=alpha,
            gamma=0.95,
            tau=0.8,      # As specified
            lr=1e-3,      # As specified  
            grad_clip=1.0
        )
        
        # Training loop
        batch_size = 128
        print(f"\nTraining for {epochs} epochs with batch size {batch_size}...", flush=True)
        start_time = time.time()
        
        best_val_loss = float('inf')
        os.makedirs('experiment', exist_ok=True)
        
        for epoch in range(epochs):
            # Training phase
            agent.q1.train()
            agent.q2.train()
            
            train_metrics = {
                'q1_loss': 0, 'q2_loss': 0,
                'cql1_loss': 0, 'cql2_loss': 0,
                'total_q1_loss': 0, 'total_q2_loss': 0
            }
            
            # Sample random batches for training
            n_batches = len(train_data['states']) // batch_size
            
            for _ in range(n_batches):
                # Get batch
                batch = pipeline.get_batch(batch_size=batch_size, split='train')
                
                # Convert to tensors
                states = torch.FloatTensor(batch['states']).to(agent.device)
                actions = torch.FloatTensor(batch['actions']).to(agent.device)
                rewards = torch.FloatTensor(batch['rewards']).to(agent.device)
                next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
                dones = torch.FloatTensor(batch['dones']).to(agent.device)
                
                # Update agent
                metrics = agent.update(states, actions, rewards, next_states, dones)
                
                # Accumulate metrics
                for key in train_metrics:
                    train_metrics[key] += metrics.get(key, 0)
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= n_batches
            
            # Validation phase
            agent.q1.eval()
            agent.q2.eval()
            
            val_q_values = []
            with torch.no_grad():
                # Sample validation batches
                for _ in range(10):  # Use 10 validation batches
                    batch = pipeline.get_batch(batch_size=batch_size, split='val')
                    
                    states = torch.FloatTensor(batch['states']).to(agent.device)
                    actions = torch.FloatTensor(batch['actions']).to(agent.device)
                    
                    # Convert continuous actions to discrete for validation
                    action_indices = []
                    for i in range(batch_size):
                        action_np = actions[i].cpu().numpy()
                        action_idx = agent.continuous_to_discrete_action(action_np)
                        action_indices.append(action_idx)
                    action_indices = torch.LongTensor(action_indices).to(agent.device)
                    
                    q1_val = agent.q1(states, action_indices).squeeze()
                    q2_val = agent.q2(states, action_indices).squeeze()
                    q_val = torch.min(q1_val, q2_val)
                    
                    val_q_values.append(q_val.mean().item())
            
            val_loss = -np.mean(val_q_values)  # Negative because we want higher Q-values
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                agent.save(f'{save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_best.pt')
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_td_loss = (train_metrics['q1_loss'] + train_metrics['q2_loss']) / 2
                print(f"Epoch {epoch+1}: "
                    f"TD Loss={avg_td_loss:.4f} (Q1={train_metrics['q1_loss']:.4f}, Q2={train_metrics['q2_loss']:.4f}), "
                    f"CQL Loss (Q1={train_metrics['cql1_loss']:.4f}, Q2={train_metrics['cql2_loss']:.4f}), "
                    f"Val Q={-val_loss:.4f}, "
                    f"Time={elapsed/60:.1f}min", flush=True)
        
        # Save final model
        agent.save(f'{save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_final.pt')

        total_time = time.time() - start_time
        print(f"\nBlock Discrete CQL ({experiment_prefix}, alpha={alpha}, bins={vp2_bins}) completed in {total_time/60:.1f} minutes!", flush=True)
        print("Models saved:", flush=True)
        print(f"  - {save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_best.pt", flush=True)
        print(f"  - {save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_final.pt", flush=True)

        return #agent, pipeline, experiment_prefix
