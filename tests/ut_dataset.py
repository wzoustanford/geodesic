from datasets import RLDataCollection 
import configs.vaso_dual_action_data_config as data_config 

def ut1_dataset():
    print(data_config)
    DC = RLDataCollection(data_config)
    for batch_idx, features in enumerate(DC.train_loader):
        print(batch_idx)
        print(features)
        if batch_idx == 3:
            break

if __name__=="__main__": 
    ut1_dataset()