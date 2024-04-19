import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


# hard coded for bipedalwalker environment, as no feature sampling needed in small obs space
class ICM(nn.Module):
    def __init__(self, obs_size, action_size, nodes = [256, 256]):
        super().__init__()
        initialLayerSize = obs_size + action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(initialLayerSize, nodes[0]),
            nn.ReLU(),
            nn.Linear(nodes[0], nodes[1]),
            nn.ReLU(),
            nn.Linear(nodes[1], obs_size)
        )
    
    def forwardModel(self, curr_state, action):

        self.currState = np.array(curr_state)
        self.actionTaken = np.array(action)
        initalLayer = np.concatenate((self.currState, self.actionTaken))

        tensorInput = torch.tensor(initalLayer)
        out = self.linear_relu_stack(tensorInput)

        self.pred_nextState = out.detach().numpy()
        return self.pred_nextState

    def inverseModel(self):
        pass

    def forwardLossAndReward(self, next_state, eita = 1, beta = 0.2):

        self.nextState = np.array(next_state)
        y = torch.tensor(self.nextState, dtype= torch.float32)
        y_pred = torch.tensor(self.pred_nextState, dtype= torch.float32)

        L_F = nn.MSELoss()
        self.forwardLoss = beta*L_F(y, y_pred).detach().numpy()
        self.intrisic_reward = eita*self.forwardLoss

        return self.forwardLoss, self.intrisic_reward
    