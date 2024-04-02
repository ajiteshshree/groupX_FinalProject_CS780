import torch
from torch import nn 
import numpy as np

class mainLoop(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forwardPass(self, currState):
        
        self.currState = np.array(currState)
        tensorInput = torch.tensor(self.currState)
        out = self.linear_relu_stack(tensorInput)

        self.actionToTake = out.detach().numpy()
        return self.actionToTake
    
    def mainLoss(self): #TO DO
        pass


