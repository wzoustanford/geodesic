import numpy as np

from datasets import SequenceDataset


def test_sequence_dataset_stores_windows_by_sequence_length_and_stride():
    dataset = SequenceDataset(seq_len=3, stride=2, capacity=10)

    for i in range(6):
        dataset.add_transition(
            state=np.array([i, i + 0.5]),
            action=np.array([i]),
            reward=float(i),
            next_state=np.array([i + 1, i + 1.5]),
            done=False,
        )

    assert len(dataset) == 2

    first_states, _, first_rewards, _, _ = dataset[0]
    second_states, _, second_rewards, _, _ = dataset[1]

    np.testing.assert_array_equal(first_states[:, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(first_rewards, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(second_states[:, 0], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(second_rewards, np.array([2.0, 3.0, 4.0]))


def test_sequence_dataset_sample_batch_shapes():
    dataset = SequenceDataset(seq_len=2, stride=1, capacity=10)

    for i in range(4):
        dataset.add_transition(
            state=np.array([i, i + 0.5]),
            action=np.array([i]),
            reward=float(i),
            next_state=np.array([i + 1, i + 1.5]),
            done=i == 3,
        )

    states, actions, rewards, next_states, dones = dataset.sample_batch(batch_size=3)

    assert states.shape == (3, 2, 2)
    assert actions.shape == (3, 2, 1)
    assert rewards.shape == (3, 2)
    assert next_states.shape == (3, 2, 2)
    assert dones.shape == (3, 2)
