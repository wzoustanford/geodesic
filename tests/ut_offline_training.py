from datasets import RLDataCollection 
from agents import MultinomialActionQLAgent
from orchestrator import Orchestrator
import configs.vaso_dual_action_data_config as data_config 

def ut_training():
    agent = MultinomialActionQLAgent(state_dim=17, a2_bins = 5)
    orc = Orchestrator(agent, 30, data_config)
    orc.start()

if __name__=="__main__": 
    ut_training()
