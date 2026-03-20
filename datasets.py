import torch 
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
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

class RLDataset(Dataset):
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

class RLDataCollection:
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

        # [extention]: distributed data ETL pre-processing using pyspark and pandas 
        train_data, val_data, test_data = self.prepare_data() 
        
        #states, actions = self.normalize_process_features(data) # [extension]: implement this later for normalization and one hot features for category features 
        train_dataset = RLDataset(train_data) 
        val_dataset = RLDataset(val_data)
        test_dataset = RLDataset(test_data)

        self.train_loader = DataLoader(train_dataset, batch_size=data_config.TRAIN_BATCH_SIZE, shuffle=data_config.SHUFFLE)
        self.val_loader = DataLoader(val_dataset, batch_size=data_config.VAL_BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=data_config.TEST_BATCH_SIZE, shuffle=True)
        
        # [online RL]: append to the lists as agent explores the environment 
    
    def get_traj_ids(self, data) -> np.ndarray:
        """Get all unique patient IDs"""
        return data[self.data_config.TRAJ_ID_COL].unique()

    def get_traj_data(self, traj_id: int, data, data_config) -> pd.DataFrame:
        """Get data for a specific patient"""
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
        print(f"Number of patients: {data[self.data_config.TRAJ_ID_COL].nunique()}")
        print(f"Columns found: {len(data.columns)} columns")
        
        # Sort by patient and time
        data = data.sort_values([self.data_config.TRAJ_ID_COL, self.data_config.TIME_COL])
        return data 
    
    def prepare_data(self):
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
            # Load training data (use all patients)
            print("1. Loading training data...")
            data = self.load_offline_trajectorys_from_file(self.data_config.COMBINED_OR_TRAIN_DATA_PATH) 
            train_traj_ids = self.get_traj_ids(data)
            print(f"   Train: {len(train_traj_ids)} patients (all from {self.data_config.COMBINED_OR_TRAIN_DATA_PATH})")

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

            print(f"   Val:   {len(val_traj_ids)} patients (from {self.data_config.EVAL_DATA_PATH})")
            print(f"   Test:  {len(test_traj_ids)} patients (from {self.data_config.EVAL_DATA_PATH})")

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

            print(f"   Train: {len(train_traj_ids)} patients")
            print(f"   Val:   {len(val_traj_ids)} patients")
            print(f"   Test:  {len(test_traj_ids)} patients")

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

    def split_data(self, patient_ids: np.ndarray,
                      train_ratio: float,
                      val_ratio: float,
                      test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split patient IDs into train/val/test sets

        Args:
            patient_ids: Array of all patient IDs
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)

        Returns:
            Tuple of (train_patients, val_patients, test_patients)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Special case: train_ratio=0.0 means we only want val and test splits
        # (used in dual-dataset mode where all train data comes from another dataset)
        if train_ratio == 0.0:
            # Split directly into val and test
            # test_ratio / (val_ratio + test_ratio) gives relative test size
            relative_test_size = test_ratio / (val_ratio + test_ratio)
            val_patients, test_patients = train_test_split(
                patient_ids,
                test_size=relative_test_size,
                random_state=self.random_seed
            )
            train_patients = np.array([])  # Empty train set
        else:
            # Standard two-step split
            # First split: train+val vs test
            train_val_patients, test_patients = train_test_split(
                patient_ids,
                test_size=test_ratio,
                random_state=self.random_seed
            )

            # Second split: train vs val
            # Calculate validation size relative to train+val
            val_size_relative = val_ratio / (train_ratio + val_ratio)

            train_patients, val_patients = train_test_split(
                train_val_patients,
                test_size=val_size_relative,
                random_state=self.random_seed
            )

        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients

        return train_patients, val_patients, test_patients

    def _build_buffer_from_split(self, traj_list: np.ndarray, state_features: list, split_name: str, data) -> Dict: 
        """
        Process a data split to create (s, a, r, s', done) tuples.
        Initial rewards are set to 0 if using learned rewards (recomputed later).
        
        Args:
            patient_list: Array of patient IDs to process
            state_features: List of state feature column names
            split_name: Name of split ('train', 'val', or 'test')
            loader: DataLoader instance to use for getting patient data
        """

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        all_patient_ids = []
        
        # Group transitions by patient for patient-aware batching
        current_idx = 0

        for traj_id in traj_list:
            traj =self.get_traj_data(traj_id, data, self.data_config)

            if len(data) < 2:  # Need at least 2 timesteps
                continue
            
            # Extract states
            states = traj[state_features].values

            actions = traj[self.data_config.ACTION_COLUMNS].values

            # Get patient mortality outcome (for manual reward if needed)
            mortality = int(traj[self.data_config.DEATH_COL].iloc[-1])

            # Store starting index for this patient
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

                all_patient_ids.append(traj_id)
                current_idx += 1

        # Convert to arrays 
        # [refactor]: consolidate the np.array and torch.FloatTensor cast and the sklearn scalar transform 
        all_states = np.array(all_states, dtype=np.float32)
        all_next_states = np.array(all_next_states, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.float32)
        all_rewards = np.array(all_rewards, dtype=np.float32)
        all_dones = np.array(all_dones, dtype=np.float32)
        all_patient_ids = np.array(all_patient_ids)

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
