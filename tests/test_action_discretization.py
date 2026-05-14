import torch

from agents import MultinomialActionQLAgent


def test_multinomial_action_transform_respects_bins_and_action_prefix(tmp_path):
    agent = MultinomialActionQLAgent(
        state_dim=3,
        a2_bins=5,
        alpha=0.0,
        device="cpu",
        save_dir=str(tmp_path),
    )
    actions = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.0, 0.5],
            [1.0, 0.0],
            [1.0, 0.499],
            [1.0, 0.5],
        ]
    )

    transformed = agent._transform_actions(actions)

    assert transformed.tolist() == [0, 1, 4, 5, 9, 9]
