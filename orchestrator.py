import torch
from datasets import RLDataset
from agents import MultinomialActionQLAgent
#import configs.vaso_single_action_data_config as data_config

class Orchestrator:
    def __init__(self, data_config, agent):
        self.dataset = RLDataset(data_config)
        self.agent = MultinomialActionQLAgent(state_dim=7, a2_bins = 5)
        
        return
    
    def start(self): 
        # ------------------------------------------------------------------
        # TODO: start()
        # ------------------------------------------------------------------
        # start orchestration, includes training. 
        # The training loop is ~130 lines duplicated almost verbatim.
        # Shared structure:
        #   1. Print header / hyperparameters
        #   2. dataset.prepare_data() → train/val/test
        for epoch in range(self.model_config.num_epochs): 
            for (index, batch) in enumerate(self.dataset.train_loader): 
                

        #   3. Epoch loop:
        #      a. Set q1/q2 to train mode
        #      b. Batch loop: sample batch → convert to tensors → self.update()
        #         → accumulate metrics
        #      c. Average metrics over batches
        #      d. Validation: set eval mode → sample val batches → compute Q-values
        #      e. Save best model if val improves
        #      f. Print progress every N epochs
        #   4. Save final model
        #
        # Differences to handle:
        #   - Discrete's train() has reward-model loading logic (should arguably live
        #     outside the agent, in a training script or Dataset subclass)
        #   - Discrete's validation converts continuous→discrete actions (handled by
        #     _transform_actions hook)
        #   - Different print messages / model naming conventions
        #     → parameterize via _agent_name() or constructor arg
        #
