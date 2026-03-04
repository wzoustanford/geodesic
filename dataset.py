from abc import ABC, abstractmethod 

class Dataset(ABC): 
    def __init__(self): 
        pass
    @abstractmethod
    def get_batch(self):
        pass 

