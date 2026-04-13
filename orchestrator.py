import torch, time, numpy as np
from datasets import OfflineRLDataCollection, SequenceDataCollection
from agents import MultinomialActionQLAgent

class Orchestrator:
    def __init__(
        self,
        agent,
        data_config,
        num_epochs=None,
        env_config=None,
        warmstart_steps=None,
    ):
        self.agent = agent
        self.data_config = data_config
        self.num_epochs = num_epochs
        self.warmstart_steps = warmstart_steps
        self.num_steps_per_epoch = getattr(data_config, 'NUM_STEPS_PER_EPOCH', None)

        if data_config.RL_MODE == "ONLINE":
            self.data_collection = SequenceDataCollection(data_config)
            self.envs = env_config.spawn()
        else:
            self.data_collection = OfflineRLDataCollection(data_config)
    
    def start(self): 
        # ------------------------------------------------------------------
        # start()
        # ------------------------------------------------------------------
        # start orchestration, includes environment play and training. 
        # Shared structure:
        #   1. Print header / hyperparameters
        start_time = time.time()
        print('-'*10 + 'starting training' + '-'*10)
        for epoch in range(self.num_epochs): 
            for batch_idx, batch in enumerate(self.data_collection.train_loader):
                if batch_idx % 100 == 0: 
                    print(str(batch_idx), end='..') 
                #print(features)
                train_metrics = self.agent.update(
                    batch[0], # states 
                    batch[1], # actions
                    batch[2], # rewards 
                    batch[3], # next_states 
                    batch[4], # dones 
                )
                # Accumulate metrics
                for key in train_metrics:
                    train_metrics[key] += train_metrics.get(key, 0)
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= len(self.data_collection.train_loader)
            
            print('\n'+'-'*5 + f'validating ... training done for epoch:{epoch}' + '-'*5)
            val_batch = next(iter(self.data_collection.val_loader))
            val_q = self.agent.validate(
                val_batch[0],
                val_batch[1],
            )
            print('<'*2 + f'validation q: {val_q}' + '<'*2)
            # Log progress 
            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_td_loss = (train_metrics['q1_loss'] + train_metrics['q2_loss']) / 2
                print(f"Epoch {epoch+1}: "
                    f"TD Loss={avg_td_loss:.4f} (Q1={train_metrics['q1_loss']:.4f}, Q2={train_metrics['q2_loss']:.4f}), "
                    #f"CQL Loss (Q1={train_metrics['cql1_loss']:.4f}, Q2={train_metrics['cql2_loss']:.4f}), "
                    f"Val Q={val_q:.4f}, "
                    f"Best Val Q={self.agent.best_val_q:.4f}, "
                    f"Time={elapsed/60:.1f}min", flush=True)
            
        # Save final model
        self.agent.save(self.agent.get_save_path('final'))
        
        total_time = time.time() - start_time
        print(f"\n({self.agent.experiment_prefix}) completed in {total_time/60:.1f} minutes!", flush=True)
        print("Models saved:", flush=True)
        print(f"  - {self.agent.get_save_path('best')}", flush=True)
        print(f"  - {self.agent.get_save_path('final')}", flush=True)

    def start_online(self):
        obs, _ = self.envs.reset()
        start_time = time.time()
        print('-'*10 + 'starting online RL training' + '-'*10)

        global_step = 0
        for epoch in range(self.num_epochs):
            for step in range(self.num_steps_per_epoch):
                if global_step < self.warmstart_steps:
                    actions = self.envs.action_space.sample()
                else:
                    actions = self.agent.sample_action(obs)

                # envs.step expects numpy arrays on CPU
                if torch.is_tensor(actions):
                    actions = actions.cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                dones = np.logical_or(terminations, truncations)

                # Use true terminal obs for buffer, not the auto-reset obs
                buffer_obs = next_obs
                if "final_obs" in infos:
                    buffer_obs = np.where(
                        dones[:, None], np.stack(infos["final_obs"]), next_obs
                    )

                self.data_collection.dataset.add_transition(obs, actions, rewards, buffer_obs, dones)
                obs = next_obs

                # Create train_loader once we have enough samples
                if self.data_collection.data_loader is None \
                        and len(self.data_collection.dataset) >= self.data_collection.batch_size:
                    self.data_collection.create_train_loader()

                if global_step >= self.warmstart_steps and self.data_collection.data_loader is not None:
                    batch = self.data_collection.sample_batch()
                    # Flatten (batch, seq_len, num_tasks, feat_dim) → (N, feat_dim)
                    # e.g. states: (32,1,10,59)→(320,59), rewards: (32,1,10)→(320,)
                    batch = tuple(
                        b.reshape(-1, *b.shape[3:]) if b.dim() > 2 else b.reshape(-1)
                        for b in batch
                    )
                    self.agent.update(*batch)

                global_step += 1

            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs}, "
                  f"global_step={global_step}, "
                  f"sequences={len(self.data_collection.dataset)}, "
                  f"time={elapsed/60:.1f}min")

        self.envs.close()
        print(f"Online training complete in {(time.time()-start_time)/60:.1f}min")
