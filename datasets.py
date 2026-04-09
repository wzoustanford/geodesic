import torch, copy
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from collections import deque
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from projects.vaso.utils import compute_vaso_clinician_rewards
from sklearn.model_selection import train_test_split
#import logging #[logging to be implemented later to replace print()]

"""
RL Dataset design principles: 
- two main parts: (1) pandas ETL processing (2) form into RL RB format and leverage torch Dataloader. The former can be parallelized 
- (1) prepare_features can be replaced by parallel processing, encode categorical features and normalize 
- (2) Store train/val/test split replay buffers with torch DataLoader, enables memory efficiency, potential support for lazy loading and parallel loading 
- easy to understand and collaborate 
- support all features for vaso RL. 
- support funcsions such as missing data check 
- Goal was not super encapsulated/oop, but get to efficient v1 by realizing above goals 
"""

class TransitionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['states'])

    def __getitem__(self, idx):
        return (
            self.data['states'][idx],
            self.data['actions'][idx],
            self.data['rewards'][idx],
            self.data['next_states'][idx],
            self.data['dones'][idx],
            self.data['n_transitions'],
            self.data['n_trajs'],
            self.data['state_features'],
        )

class SequenceDataset(Dataset):
    def __init__(self, seq_len, stride=None, capacity=None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self._buf_s, self._buf_a, self._buf_r = [], [], []
        self._buf_ns, self._buf_d = [], []
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def add_transition(self, state, action, reward, next_state, done):
        self._buf_s.append(np.asarray(state, dtype=np.float32))
        self._buf_a.append(np.asarray(action, dtype=np.float32))
        self._buf_r.append(np.asarray(reward, dtype=np.float32))
        self._buf_ns.append(np.asarray(next_state, dtype=np.float32))
        self._buf_d.append(np.asarray(done, dtype=np.float32))
        if len(self._buf_s) >= self.seq_len:
            if (len(self._buf_s) - self.seq_len) % self.stride == 0:
                self._store_window(len(self._buf_s) - self.seq_len)

    def _store_window(self, start):
        end = start + self.seq_len
        self.states.append(np.stack(self._buf_s[start:end]))
        self.actions.append(np.stack(self._buf_a[start:end]))
        self.rewards.append(np.array(self._buf_r[start:end]))
        self.next_states.append(np.stack(self._buf_ns[start:end]))
        self.dones.append(np.array(self._buf_d[start:end]))

    def _clear_active(self):
        self._buf_s, self._buf_a, self._buf_r = [], [], []
        self._buf_ns, self._buf_d = [], []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

class PrioritySampler(torch.utils.data.Sampler):
    """Implements a sum-tree algorithm for prioritized experience replay."""
    # [note: to re-check and test thoroughly, AI implementation placeholder only]
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.size = 0
        self.max_priority = 1.0

    # [note: to re-check and test thoroughly, AI implementation placeholder only]
    def __iter__(self):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, size=self.size, p=probs, replace=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.size

    # [note: to re-check and test thoroughly, AI implementation placeholder only]
    def update(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = abs(p) + self.epsilon
        self.max_priority = max(self.max_priority, self.priorities[:self.size].max())

    # [note: to re-check and test thoroughly, AI implementation placeholder only]
    def weights(self, indices):
        probs = self.priorities[indices] ** self.alpha
        probs /= (self.priorities[:self.size] ** self.alpha).sum()
        w = (self.size * probs) ** (-self.beta)
        return w / w.max()

    # [note: to re-check and test thoroughly, AI implementation placeholder only]
    def add(self, count=1):
        for _ in range(count):
            if self.size < self.capacity:
                self.priorities[self.size] = self.max_priority
                self.size += 1

class SequenceDataCollection:
    def __init__(self, data_config):
        self.dataset = SequenceDataset(
            seq_len=data_config.SEQ_LEN,
            stride=data_config.STRIDE,
            capacity=data_config.BUFFER_CAPACITY,
        )
        self.batch_size = data_config.TRAIN_BATCH_SIZE
        self.shuffle = getattr(data_config, 'SHUFFLE', False)
        self.data_loader = None

    def create_train_loader(self):
        assert len(self.dataset) >= self.batch_size, \
            f"Not enough samples ({len(self.dataset)}) for batch_size ({self.batch_size})"
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self._iter = iter(self.data_loader)

    def sample_batch(self):
        # Note on deque eviction and iterator staleness:
        # SequenceDataset stores data in deque(maxlen=capacity). The DataLoader
        # iterator snapshots len(dataset) at creation and builds a shuffled index
        # permutation. If items are evicted from the deque between next() calls:
        #   1. deque has [A, B, C, D, E] (indices 0-4)
        #   2. iter(loader) shuffles indices [0,1,2,3,4] -> e.g. [3,1,4,0,2]
        #   3. next() -> dataset[3] -> returns D
        #   4. deque full, new item added -> A popped, F appended -> [B,C,D,E,F]
        #   5. next() -> dataset[1] -> returns C (was B before pop, indices shifted)
        # This means some items may be skipped or sampled twice. For RL replay
        # buffers this bias is negligible since the buffer changes slowly relative
        # to iteration speed. The iterator is refreshed when exhausted.
        assert self.data_loader is not None, \
            "Call create_train_loader() before sampling"
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.data_loader)
            return next(self._iter)

class OfflineRLDataCollection:
    def __init__(self, data_config, reward_model=None): 
        # data config: configuration file for data 
        # from_csv_file: the csv file containing offline RL data 
        # reward_source: for IRL research, which rewards source, manual defined or learned reward model 
        
        self.data_config = data_config #pandas dataframe 

        if reward_model is None: 
            self.reward_model = self.data_config.REWARD_MODEL 
        else: 
            self.reward_model = reward_model 
        
        self.scaler = StandardScaler()
        
        self.random_seed = data_config.RANDOM_SEED 

        rewards = None # to implement for cases where offline RL data includes rewards 

        online = True if hasattr(data_config, "RL_MODE") and data_config.RL_MODE == "ONLINE" else False 

        if online: 
            # [online RL]: append to the lists as agent explores the environment 
            self.online_data = self.create_empty_dataset(data_config.STATE_COLUMNS)
            self.online_reply_buffer = None 
        else: 
            # [extention]: distributed data ETL pre-processing using pyspark and pandas 
            train_data, val_data, test_data = self.prepare_offline_data() 
            
            #states, actions = self.normalize_process_features(data) # [extension]: implement this later for normalization and one hot features for category features 
            train_dataset = TransitionDataset(train_data) 
            val_dataset = TransitionDataset(val_data)
            test_dataset = TransitionDataset(test_data)

            self.train_loader = DataLoader(train_dataset, batch_size=data_config.TRAIN_BATCH_SIZE, shuffle=data_config.SHUFFLE)
            self.val_loader = DataLoader(val_dataset, batch_size=data_config.VAL_BATCH_SIZE, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=data_config.TEST_BATCH_SIZE, shuffle=True)
    
    def get_traj_ids(self, data) -> np.ndarray:
        """Get all unique trajectory IDs"""
        return data[self.data_config.TRAJ_ID_COL].unique()

    def get_traj_data(self, traj_id: int, data, data_config) -> pd.DataFrame:
        """Get data for a specific trajectory"""
        if data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return data[data[data_config.TRAJ_ID_COL] == traj_id].copy()

    def load_offline_trajectorys_from_file(self, filepath) -> pd.DataFrame:
        """
        Load data from CSV file 
        Assume this can be done on one machine with sufficiently large RAM. 
        If not, write a map-reduce sharding job 

        Returns: 
            Loaded DataFrame 
        """

        print(f"Loading data from {filepath}...")
        
        data = pd.read_csv(filepath)
        
        print(f"Loaded {len(data)} records")
        print(f"Number of trajectorys: {data[self.data_config.TRAJ_ID_COL].nunique()}")
        print(f"Columns found: {len(data.columns)} columns")
        
        # Sort by trajectory and time
        data = data.sort_values([self.data_config.TRAJ_ID_COL, self.data_config.TIME_COL])
        return data 
    
    def create_loader_for_online_replay_buffer(self): 
        self.online_data_copy = copy.deepcopy(self.online_data)
        self.online_replay_buffer = DataLoader(TransitionDataset(self.online_data_copy), batch_size = self.data_config.TRAIN_BATCH_SIZE, shuffle=False)
    
    def create_empty_dataset(self, state_features: list) -> Dict:
        """Create an empty replay buffer dict matching the TransitionDataset schema."""
        return {
            'states': np.empty((0, len(state_features)), dtype=np.float32),
            'actions': np.empty((0, 1), dtype=np.float32),
            'rewards': np.empty((0,), dtype=np.float32),
            'next_states': np.empty((0, len(state_features)), dtype=np.float32),
            'dones': np.empty((0,), dtype=np.float32),
            'n_transitions': 0,
            'n_trajs': 0,
            'state_features': state_features,
        }

    def add_transition(self, state, action, reward: float, next_state, done: float) -> Dict:
        """Append a single (s, a, r, s', done) transition to an existing dataset dict."""
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, -1)
        action = np.array(action, dtype=np.float32).reshape(1, -1)

        self.online_data['states'] = np.concatenate([self.online_data['states'], state], axis=0)
        self.online_data['actions'] = np.concatenate([self.online_data['actions'], action], axis=0)
        self.online_data['rewards'] = np.append(self.online_data['rewards'], reward)
        self.online_data['next_states'] = np.concatenate([self.online_data['next_states'], next_state], axis=0)
        self.online_data['dones'] = np.append(self.online_data['dones'], done)
        self.online_data['n_transitions'] = len(self.online_data['states'])
    

    def prepare_offline_data(self): 
        """
        Complete data preparation pipeline producing (s, a, r, s', done) tuples.
        Uses learned rewards if reward_source='learned' and a model is loaded.

        In single-dataset mode:
            - Splits combined_or_train_data_path into train/val/test (70/15/15)

        In dual-dataset mode:
            - All trajectories from combined_or_train_data_path used for training
            - Trajectories from eval_data_path split into val/test (50/50)
        """

        print(f"Preparing RL data ...")
        print(f"Reward source: {self.data_config.REWARD_MODEL}")
        
        if self.data_config.DUAL_DATASET_MODE:
            print(f"Dataset mode: DUAL-DATASET")
            print(f"  Train data: {self.data_config.COMBINED_OR_TRAIN_DATA_PATH}")
            print(f"  Eval data:  {self.data_config.EVAL_DATA_PATH}")
        else:
            print(f"Dataset mode: SINGLE-DATASET")
            print(f"  Data path: {self.data_config.COMBINED_OR_TRAIN_DATA_PATH}")
        print("="*60)

        if self.data_config.DUAL_DATASET_MODE:
            # DUAL-DATASET MODE
            # Load training data (use all trajectorys)
            print("1. Loading training data...")
            data = self.load_offline_trajectorys_from_file(self.data_config.COMBINED_OR_TRAIN_DATA_PATH) 
            train_traj_ids = self.get_traj_ids(data)
            print(f"   Train: {len(train_traj_ids)} trajectorys (all from {self.data_config.COMBINED_OR_TRAIN_DATA_PATH})")

            # Load evaluation data (split into val/test)
            print("2. Loading evaluation data...")
            eval_data = self.load_offline_trajectorys_from_file(self.data_config.EVAL_DATA_PATH)
            eval_traj_ids = self.get_traj_ids(eval_data)

            # Split eval data 50/50 into val and test
            val_traj_ids, test_traj_ids = self.split_data(
                eval_traj_ids,
                train_ratio=0.0,
                val_ratio=0.5,
                test_ratio=0.5
            )[1:]  # Skip empty train split

            print(f"   Val:   {len(val_traj_ids)} trajectorys (from {self.data_config.EVAL_DATA_PATH})")
            print(f"   Test:  {len(test_traj_ids)} trajectorys (from {self.data_config.EVAL_DATA_PATH})")

            # Encode categorical features for both loaders
            print("3. Encoding categorical features...")
            train_data = self.encode_categorical_features(train_data)
            eval_data= self.encode_categorical_features(eval_data)

            # Process each split
            print("4. Processing transitions...")
            train_data = self._build_buffer_from_split(train_traj_ids, self.data_config.STATE_COLUMNS, 'train', train_data)
            val_data = self._build_buffer_from_split(val_traj_ids, self.data_config.STATE_COLUMNS, 'val', eval_data)
            test_data = self._build_buffer_from_split(test_traj_ids, self.data_config.STATE_COLUMNS, 'test', eval_data)
        else:
            # SINGLE-DATASET MODE (original behavior)
            print("1. Loading and splitting data...")
            data = self.load_offline_trajectorys_from_file(self.data_config.COMBINED_OR_TRAIN_DATA_PATH)
            traj_ids = self.get_traj_ids(data)
            train_traj_ids, val_traj_ids, test_traj_ids = self.split_data(traj_ids, self.data_config.TRAIN_RATIO, self.data_config.VAL_RATIO, self.data_config.TEST_RATIO)

            print(f"   Train: {len(train_traj_ids)} trajectorys")
            print(f"   Val:   {len(val_traj_ids)} trajectorys")
            print(f"   Test:  {len(test_traj_ids)} trajectorys")

            # Encode categorical features
            print("2. Encoding categorical features...")
            data = self.encode_categorical_features(data)
            
            # Process each split separately
            print("3. Processing transitions...")
            train_data = self._build_buffer_from_split(train_traj_ids, self.data_config.STATE_COLUMNS, 'train', data)
            val_data = self._build_buffer_from_split(val_traj_ids, self.data_config.STATE_COLUMNS, 'val', data)
            test_data = self._build_buffer_from_split(test_traj_ids, self.data_config.STATE_COLUMNS, 'test', data)

        # If using learned rewards, recompute rewards now that data is normalized 
        if False: # [implement IRL learned rewards later] self.reward_model != 'manual': 
            step_num = "5" if self.data_config.DUAL_DATASET_MODE else "4" 
            print(f"{step_num}. Computing learned rewards...") 
            self._recompute_learned_rewards() 

        print("\n Data pipeline complete!")

        return train_data, val_data, test_data

    def split_data(self, trajectory_ids: np.ndarray,
                      train_ratio: float,
                      val_ratio: float,
                      test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split trajectory IDs into train/val/test sets

        Args:
            trajectory_ids: Array of all trajectory IDs
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)

        Returns:
            Tuple of (train_trajectorys, val_trajectorys, test_trajectorys)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Special case: train_ratio=0.0 means we only want val and test splits
        # (used in dual-dataset mode where all train data comes from another dataset)
        if train_ratio == 0.0:
            # Split directly into val and test
            # test_ratio / (val_ratio + test_ratio) gives relative test size
            relative_test_size = test_ratio / (val_ratio + test_ratio)
            val_trajectorys, test_trajectorys = train_test_split(
                trajectory_ids,
                test_size=relative_test_size,
                random_state=self.random_seed
            )
            train_trajectorys = np.array([])  # Empty train set
        else:
            # Standard two-step split
            # First split: train+val vs test
            train_val_trajectorys, test_trajectorys = train_test_split(
                trajectory_ids,
                test_size=test_ratio,
                random_state=self.random_seed
            )

            # Second split: train vs val
            # Calculate validation size relative to train+val
            val_size_relative = val_ratio / (train_ratio + val_ratio)

            train_trajectorys, val_trajectorys = train_test_split(
                train_val_trajectorys,
                test_size=val_size_relative,
                random_state=self.random_seed
            )

        self.train_trajectorys = train_trajectorys
        self.val_trajectorys = val_trajectorys
        self.test_trajectorys = test_trajectorys

        return train_trajectorys, val_trajectorys, test_trajectorys

    def _build_buffer_from_split(self, traj_list: np.ndarray, state_features: list, split_name: str, data) -> Dict: 
        """
        Process a data split to create (s, a, r, s', done) tuples.
        Initial rewards are set to 0 if using learned rewards (recomputed later).
        
        Args:
            trajectory_list: Array of trajectory IDs to process
            state_features: List of state feature column names
            split_name: Name of split ('train', 'val', or 'test')
            loader: DataLoader instance to use for getting trajectory data
        """

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        all_trajectory_ids = []
        
        # Group transitions by trajectory for trajectory-aware batching
        current_idx = 0

        for traj_id in traj_list:
            traj =self.get_traj_data(traj_id, data, self.data_config)

            if len(data) < 2:  # Need at least 2 timesteps
                continue
            
            # Extract states
            states = traj[state_features].values

            actions = traj[self.data_config.ACTION_COLUMNS].values

            # Get trajectory outcome (for manual reward if needed)
            mortality = int(traj[self.data_config.DEATH_COL].iloc[-1])

            # Store starting index for this trajectory
            traj_start_idx = current_idx

            # Create transitions
            for t in range(len(states) - 1):
                all_states.append(states[t])
                all_next_states.append(states[t + 1])
                all_actions.append(actions[t])

                # Check if terminal
                is_terminal = (t == len(states) - 2)
                all_dones.append(1.0 if is_terminal else 0.0)

                # Initial reward (will be recomputed if using learned rewards)
                if self.data_config.REWARD_MODEL == 'manual':
                    # Use manual reward
                    reward = compute_vaso_clinician_rewards(
                        states, actions, t,
                        is_terminal, mortality, state_features
                    )
                elif self.data_config.REWARD_MODEL == 'mortality_only':
                    # Mortality-only reward: 1 for death at terminal, 0 otherwise
                    # This is a sparse binary signal at the end of trajectory
                    if is_terminal and mortality == 1:
                        reward = 1.0  # Death penalty signal
                    else:
                        reward = 0.0
                else:
                    if self.reward_combine_lambda is not None:
                        # Compute manual reward for later combination with IRL reward
                        reward = compute_vaso_clinician_rewards(
                            states, actions, t,
                            is_terminal, mortality, state_features
                        )
                    else:
                        # Placeholder - will be replaced by IRL reward after normalization
                        reward = 0.0

                all_rewards.append(reward)

                all_trajectory_ids.append(traj_id)
                current_idx += 1

        # Convert to arrays 
        # [refactor]: consolidate the np.array and torch.FloatTensor cast and the sklearn scalar transform 
        all_states = np.array(all_states, dtype=np.float32)
        all_next_states = np.array(all_next_states, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.float32)
        all_rewards = np.array(all_rewards, dtype=np.float32)
        all_dones = np.array(all_dones, dtype=np.float32)
        all_trajectory_ids = np.array(all_trajectory_ids)

        # Normalize states (fit scaler on train only)
        if split_name == 'train':
            all_states_norm = self.scaler.fit_transform(all_states)
        else:
            all_states_norm = self.scaler.transform(all_states)
        all_next_states_norm = self.scaler.transform(all_next_states)


        print(f"   {split_name}: {len(all_states)} transitions ")

        return {
            'states': all_states_norm,
            'actions': all_actions,
            'rewards': all_rewards,
            'next_states': all_next_states_norm,
            'dones': all_dones,
            'n_transitions': len(all_states),
            'n_trajs': len(traj_list),
            'state_features': state_features
        }

    """
    Auxiliary class methods  
    """
    def check_missing_data(self, data, features: List[str]) -> bool:
        """
        Simple check for missing data - just reports what's missing
        
        Args:
            features: List of feature columns to check
            
        Returns:
            True if no missing data, False otherwise
        """

        print("\n" + "="*60)
        print("MISSING DATA CHECK")
        print("="*60)
        
        all_good = True
        not_found = []
        has_nulls = []
        
        for feature in features:
            if feature not in data.columns:
                print(f"Feature '{feature}' NOT FOUND in data")
                not_found.append(feature)
                all_good = False
            else:
                missing_count = data[feature].isnull().sum()
                if missing_count > 0:
                    pct = (missing_count / len(data)) * 100
                    print(f" {feature}: {missing_count} missing ({pct:.2f}%)")
                    has_nulls.append((feature, missing_count, pct))
                    all_good = False
                else:
                    print(f"✓  {feature}: OK")
        
        # Summary
        print("-"*60)
        if not_found:
            print(f"\n{len(not_found)} features NOT FOUND:")
            for f in not_found:
                print(f"   - {f}")
        
        if has_nulls:
            print(f"\n{len(has_nulls)} features have MISSING DATA:")
            for f, count, pct in has_nulls:
                print(f"   - {f}: {count} missing ({pct:.2f}%)")
        
        if all_good:
            print("\n All features present with no missing data!")
        else:
            print("\n Data issues found - cannot proceed with training")
        
        return all_good

    def encode_categorical_features(self, data) -> pd.DataFrame:
        """
        Encode categorical string features (e.g. gender, ethnicity) to numbers
        Ensures reproducibility by sorting unique values before encoding
        """

        for feature in self.data_config.CATEGORICAL_FEATURES:
            if feature in data.columns:
                # Convert to string first to handle any mixed types
                str_values = data[feature].astype(str)
                
                # Get unique values and sort them for reproducibility
                unique_values = sorted(str_values.unique())
                
                # Create mapping dictionary
                value_to_int = {val: i for i, val in enumerate(unique_values)}
                
                # Apply mapping
                data[feature] = str_values.map(value_to_int)
                print(f"Encoded {feature}: {unique_values} → {list(range(len(unique_values)))}")
        return data 

    # [extention]: to implement later for adding one-hot categorical features 
    def normalize_process_features(self, data) -> None:
        """
        Prepare features for model training
        
        Args:
            model_type: 'binary' or 'dual' for different feature sets
            
        Returns:
            Tuple of (normalized_features, actions, feature_names)
        """

        print(f"\nPreparing features...")
        print(f"State columns ({len(self.data_config.STATE_COLUMNS )}): {self.data_config.STATE_COLUMNS }")
        print(f"Action columns: {self.data_config.ACTION_COLUMNS }")
        
        # Check for missing data
        all_columns = self.data_config.STATE_COLUMNS + self.data_config.ACTION_COLUMNS 
        if not self.check_missing_data(all_columns):
            raise ValueError("Cannot prepare features due to missing data!")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        self.encode_categorical_features()
        
        # Normalize state features
        S = data[self.data_config.STATE_COLUMNS].values
        states = self.scaler.fit_transform(S)
        
        actions = data[self.data_config.ACTION_COLUMNS ].values
        
        print(f"\n Data prepared successfully!")
        print(f"  State shape: {self.states.shape}")
        print(f"  Action shape: {self.actions.shape}")
        
        return states, actions
