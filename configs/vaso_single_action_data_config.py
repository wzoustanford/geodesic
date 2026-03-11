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

# ==================== Preprocessing Parameters ====================

# Categorical features that need encoding
CATEGORICAL_FEATURES = ['ethnicity', 'gender']

# ==================== Data Path ====================

DATA_PATH = 'sample_data_oviss.csv'
