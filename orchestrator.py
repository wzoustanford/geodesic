import torch, time
from datasets import RLDataCollection
from agents import MultinomialActionQLAgent

class Orchestrator:
    def __init__(self, agent , num_epochs, data_config):
        self.data_collection = RLDataCollection(data_config)
        self.agent = agent
        
        self.num_epochs = num_epochs
    
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
            print('<'*2 + 'validation loss: {val_loss}' + '<'*2)
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
