import torch 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List 
import numpy as np 
from sklearn.preprocessing import StandardScaler

#import logging #[logging to be implemented later to replace print()]

class TransitionDataset(Dataset):
    def __init__(self, data_config, from_csv_file=None):
        self.data_config = data_config
        self.scalar = StandardScaler()
        
        if from_csv_file is not None: 
            # offline RL: loading data from CSV 
            self.data = self.load_offline_trajectorys_from_file(data_config)

            # [extention]: distributed data ETL pre-processing using pyspark and pandas 
            states, actions = self.prepare_features()


        # [online RL]: append to the lists as agent explores the environment 

    def __len__(self):
        return len(self.state_sqs)

    def __getitem__(self, idx): 
        return [self.state_sqs[idx], self.action_sqs[idx], self.reward_sqs[idx]]

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training
        
        Args:
            model_type: 'binary' or 'dual' for different feature sets
            
        Returns:
            Tuple of (normalized_features, actions, feature_names)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        state_columns = self.data_config.STATE_COLUMNS 
        action_columns = self.data_config.ACTION_COLUMNS 


        print(f"\nPreparing {self.model_type.upper()} features...")
        print(f"State columns ({len(state_columns)}): {state_columns}")
        print(f"Action columns: {action_columns}")
        
        # Check for missing data
        all_columns = state_columns + action_columns
        if not self.check_missing_data(all_columns):
            raise ValueError("Cannot prepare features due to missing data!")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        self.encode_categorical_features()
        
        # Extract and normalize state features
        S = self.data[state_columns].values
        S_normalized = self.scaler.fit_transform(S)
        
        a = self.data[action_columns].values
        
        print(f"\n Data prepared successfully!")
        print(f"  State shape: {S_normalized.shape}")
        print(f"  Action shape: {a.shape}")
        
        return S_normalized, a

    def load_offline_trajectorys_from_file(self) -> pd.DataFrame:
        """
        Load data from CSV file 
        Assume this can be done on one machine with sufficiently large RAM. 
        If not, write a map-reduce sharding job 

        Returns: 
            Loaded DataFrame 
        """

        if self.verbose:
            print(f"Loading data from {self.data_config.DATA_PATH}...")
        
        self.data = pd.read_csv(self.data_config.DATA_PATH)
        
        print(f"Loaded {len(self.data)} records")
        print(f"Number of patients: {self.data[self.data_config.TRAJ_ID_COL].nunique()}")
        print(f"Columns found: {len(self.data.columns)} columns")
        
        # Sort by patient and time
        self.data = self.data.sort_values([self.data_config.TRAJ_ID_COL, self.data_config.TIME_COL])
        

    def check_missing_data(self, features: List[str]) -> bool:
        """
        Simple check for missing data - just reports what's missing
        
        Args:
            features: List of feature columns to check
            
        Returns:
            True if no missing data, False otherwise
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("MISSING DATA CHECK")
        print("="*60)
        
        all_good = True
        not_found = []
        has_nulls = []
        
        for feature in features:
            if feature not in self.data.columns:
                print(f"Feature '{feature}' NOT FOUND in data")
                not_found.append(feature)
                all_good = False
            else:
                missing_count = self.data[feature].isnull().sum()
                if missing_count > 0:
                    pct = (missing_count / len(self.data)) * 100
                    print(f" {feature}: {missing_count} missing ({pct:.2f}%)")
                    has_nulls.append((feature, missing_count, pct))
                    all_good = False
                else:
                    if self.verbose:
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

    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical string features (e.g. gender, ethnicity) to numbers
        Ensures reproducibility by sorting unique values before encoding
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        for feature in self.data_config.CATEGORICAL_FEATURES:
            if feature in self.data.columns:
                # Convert to string first to handle any mixed types
                str_values = self.data[feature].astype(str)
                
                # Get unique values and sort them for reproducibility
                unique_values = sorted(str_values.unique())
                
                # Create mapping dictionary
                value_to_int = {val: i for i, val in enumerate(unique_values)}
                
                # Apply mapping
                self.data[feature] = str_values.map(value_to_int)
                print(f"Encoded {feature}: {unique_values} → {list(range(len(unique_values)))}")
