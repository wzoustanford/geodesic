import numpy as np 


def compute_vaso_clinician_rewards(
    states: np.ndarray,
    actions: np.ndarray,
    current_timestep: int,
    is_terminal: bool,
    mortality: int,
    state_features: list
) -> float:
    """
    Compute reward based on clinical outcomes

    Args:
        states: All states for this patient (T x num_features)
        actions: All actions for this patient (T,) or (T x num_actions)
        current_timestep: Current timestep index
        is_terminal: Whether this is the last timestep
        mortality: Whether patient died (1) or survived (0)
        state_features: List of feature names for indexing

    Returns:
        Reward value
    """
    # 1. Base reward for survival at each time point
    reward = 1.0

    # Get indices for key features
    mbp_idx = state_features.index('mbp') if 'mbp' in state_features else None
    lactate_idx = state_features.index('lactate') if 'lactate' in state_features else None
    sofa_idx = state_features.index('sofa') if 'sofa' in state_features else None
    norepi_idx = state_features.index('norepinephrine') if 'norepinephrine' in state_features else None

    state = states[current_timestep]

    # 3. Benefit for improved clinical parameters

    # 3a. Decreased lactate levels (check next 6 hours)
    if lactate_idx is not None:
        lactate_current = state[lactate_idx]
        # Look ahead up to 6 hours (or until end of trajectory)
        lookback_6hr = min(current_timestep + 6, len(states) - 1)
        if lookback_6hr > current_timestep:
            lactate_future = states[lookback_6hr][lactate_idx]
            if lactate_future < lactate_current:
                reward += 1.0  # Improved metabolic status

    # 3b. Increased MBP to >= 65 mmHg (check next 4 hours)
    if mbp_idx is not None:
        mbp_current = state[mbp_idx]
        # Only reward if MBP was initially below 65
        if mbp_current < 65:
            # Look ahead up to 4 hours (or until end of trajectory)
            lookback_4hr = min(current_timestep + 4, len(states) - 1)
            if lookback_4hr > current_timestep:
                mbp_future = states[lookback_4hr][mbp_idx]
                if mbp_future >= 65:
                    reward += 1.0  # Improved hemodynamic stability

    # 3c. Decreased SOFA score (check next 6 hours)
    if sofa_idx is not None:
        sofa_current = state[sofa_idx]
        # Look ahead up to 6 hours (or until end of trajectory)
        lookback_6hr = min(current_timestep + 6, len(states) - 1)
        if lookback_6hr > current_timestep:
            sofa_future = states[lookback_6hr][sofa_idx]
            if sofa_future < sofa_current:
                reward += 3.0  # Improved organ function

    # 3d. Decreased norepinephrine usage (check next 4 hours)
    # Norepinephrine can be in state (Binary CQL) or action (Dual CQL)
    if norepi_idx is not None:
        # Binary CQL: norepinephrine is in state
        norepi_current = state[norepi_idx]
        lookback_4hr = min(current_timestep + 4, len(states) - 1)
        if lookback_4hr > current_timestep:
            norepi_future = states[lookback_4hr][norepi_idx]
            if norepi_future < norepi_current:
                reward += 1.0  # Improved cardiovascular status
    elif actions.ndim > 1 and actions.shape[1] >= 2:
        # Dual CQL: norepinephrine is the second action
        norepi_current = actions[current_timestep][1]
        lookback_4hr = min(current_timestep + 4, len(actions) - 1)
        if lookback_4hr > current_timestep:
            norepi_future = actions[lookback_4hr][1]
            if norepi_future < norepi_current:
                reward += 1.0  # Improved cardiovascular status

    # 2. Penalty for death in the last state
    if is_terminal:
        if mortality == 1:
            reward -= 20.0  # Death penalty
        # Note: Survival is already rewarded through base +1.0 at each timestep

    return reward
