"""
Configuration file for data processing and feature definitions
"""

# ==================== Feature Definitions ====================

# state columns 
STATE_COLUMNS = [
    'time_hour',      # Time since ICU admission
    'mbp',            # Mean blood pressure
    'lactate',        # Lactate level
    'bun',            # Blood urea nitrogen
    'creatinine',     # Creatinine level
    'fluid',          # Fluid intake
    'total_fluid',    # Total fluid balance
    'uo_h',           # Urine output per hour
    'ventil',         # Ventilation status (binary)
    'rrt',            # Renal replacement therapy (binary)
    'sofa',           # Sequential Organ Failure Assessment score
    'cortico',        # Corticosteroid use (binary)
    'height',         # Patient height
    'weight',         # Patient weight
    'ethnicity',      # Patient ethnicity (categorical)
    'age',            # Patient age
    'gender',         # Patient gender (categorical)
    'norepinephrine'  # Norepinephrine dose (0-0.5 mcg/kg/min)
]

# actions columns 
ACTION_COLUMNS = [
    'action_vaso',      # Vasopressin dose (normalized 0-1)    
]

ACTION_TYPES = [
    'binary',
]

# ==================== Data Processing Parameters ====================

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Patient ID column
TRAJ_ID_COL = 'subject_id'

# Time column
TIME_COL = 'time_hour'

# Outcome columns
DEATH_COL = 'death'
OPTIMAL_ACTION_COL = 'optimal_action'

# ==================== Preprocessing Parameters ====================

# Categorical features that need encoding
CATEGORICAL_FEATURES = ['ethnicity', 'gender']

# Binary features (already 0/1)
BINARY_FEATURES = ['ventil', 'rrt', 'cortico']

# Continuous features that need normalization
CONTINUOUS_FEATURES = [
    'time_hour', 'mbp', 'lactate', 'bun', 'creatinine',
    'fluid', 'total_fluid', 'uo_h', 'sofa', 'height',
    'weight', 'age', 'norepinephrine'
]

# ==================== Missing Data Handling ====================

# NOTE: The data should be pre-processed with forward-fill for missing values
# If any missing values are found, the loader will:
# 1. Report detailed statistics about missing values
# 2. Stop execution and require manual intervention

# Set to True to allow automatic filling (NOT RECOMMENDED)
ALLOW_AUTO_FILL = False

# Set to True to get detailed missing data report
VERBOSE_MISSING_DATA_REPORT = True

# Maximum allowed percentage of missing data per feature
MAX_MISSING_PERCENTAGE = 0.0  # 0% - we expect no missing data

# Default values for missing features (ONLY used if ALLOW_AUTO_FILL is True)
# These should NOT be used in production - data should be properly preprocessed
DEFAULT_VALUES = {
    'age': 65,
    'gender': 0,
    'ethnicity': 0,
    'height': 170,
    'weight': 75,
    'time_hour': 0,
    'mbp': 70,
    'lactate': 2.0,
    'bun': 20,
    'creatinine': 1.0,
    'fluid': 0,
    'total_fluid': 0,
    'uo_h': 50,
    'ventil': 0,
    'rrt': 0,
    'sofa': 8,
    'cortico': 0,
    'norepinephrine': 0.1,
    'action_vaso': 0
}

# ==================== Model Parameters ====================

# Q-Network architecture
Q_NETWORK_HIDDEN_DIMS = [256, 128, 64]

# CQL specific parameters
CQL_ALPHA = 0.001  # Conservative penalty strength
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
NUM_EPOCHS = 100

# Action sampling for continuous CQL
NUM_ACTION_SAMPLES = 100  # Number of samples for action selection

# ==================== File Paths ====================

DATA_PATH = 'sample_data_oviss.csv'
CHECKPOINT_DIR = 'experiment/'
FIGURE_DIR = 'experiment/figures/'
