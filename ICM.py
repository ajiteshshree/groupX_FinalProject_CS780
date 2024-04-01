import torch
from torch import nn
import math
import numpy as np


# hard coded for bipedalwalker environment, as no feature sampling needed in small obs space
class ICM(nn.Module):
    def __init__(self, obs_size, action_size, nodes = [256, 256]):
        initialLayerSize = obs_size + action_size
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(initialLayerSize, nodes[0]),
            nn.ReLU(),
            nn.Linear(nodes[0], nodes[1]),
            nn.ReLU(),
            nn.Linear(nodes[1], obs_size),
        )
    
    def forwardModel(self, curr_state, action):

        self.currState = np.array(curr_state)
        self.actionTaken = np.array(action)
        initalLayer = np.concatenate((self.currState, self.actionTaken))

        tensorInput = torch.tensor(initalLayer)
        out = self.linear_relu_stack(tensorInput)

        self.pred_nextState = out.detach().numpy()
        return self.pred_nextState

    def forwardLossAndReward(self, next_state, eita = 1):

        self.nextState = np.array(next_state)

        self.forwardLoss = 0.5*math.pow(math.dist(self.nextState, self.pred_nextState), 2)
        self.intrisic_reward = eita*self.forwardLoss

        return self.forwardLoss, self.intrisic_reward
    