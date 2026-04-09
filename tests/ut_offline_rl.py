from datasets import OfflineRLDataCollection 
from agents import MultinomialActionQLAgent
from orchestrator import Orchestrator
import configs.vaso_dual_action_data_config as data_config 

def ut_training():
    num_epochs = 30
    agent = MultinomialActionQLAgent(state_dim=17, a2_bins = 5)
    orc = Orchestrator(agent, num_epochs, data_config)
    orc.start()

if __name__=="__main__": 
    ut_training()
