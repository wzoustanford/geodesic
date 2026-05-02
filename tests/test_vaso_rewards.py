import numpy as np

from projects.vaso.utils import compute_vaso_clinician_rewards


def test_reward_returns_base_survival_reward_without_changes():
    features = ["mbp", "lactate", "sofa", "norepinephrine"]
    states = np.array([[70.0, 2.0, 5.0, 0.2]])
    actions = np.zeros((1, 2))

    reward = compute_vaso_clinician_rewards(
        states,
        actions,
        current_timestep=0,
        is_terminal=False,
        mortality=0,
        state_features=features,
    )

    assert reward == 1.0


def test_reward_adds_clinical_improvement_terms():
    features = ["mbp", "lactate", "sofa", "norepinephrine"]
    states = np.array(
        [
            [60.0, 4.0, 8.0, 0.4],
            [60.0, 4.0, 8.0, 0.4],
            [60.0, 4.0, 8.0, 0.4],
            [60.0, 4.0, 8.0, 0.4],
            [70.0, 4.0, 8.0, 0.1],
            [70.0, 4.0, 8.0, 0.1],
            [70.0, 2.0, 6.0, 0.1],
        ]
    )
    actions = np.zeros((len(states), 2))

    reward = compute_vaso_clinician_rewards(
        states,
        actions,
        current_timestep=0,
        is_terminal=False,
        mortality=0,
        state_features=features,
    )

    assert reward == 7.0


def test_reward_applies_terminal_death_penalty():
    features = ["mbp", "lactate", "sofa"]
    states = np.array([[70.0, 2.0, 5.0]])
    actions = np.zeros((1, 1))

    reward = compute_vaso_clinician_rewards(
        states,
        actions,
        current_timestep=0,
        is_terminal=True,
        mortality=1,
        state_features=features,
    )

    assert reward == -19.0
