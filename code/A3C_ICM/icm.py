import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=2, alpha=1, beta=0.2):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        input = input_dims[0]
        self.inverse = nn.Linear(input*2, 256)
        self.pi_logits = nn.Linear(256, n_actions)

        self.dense1 = nn.Linear(input+1, 256)
        self.new_state = nn.Linear(256, input)

        device = T.device('cpu')
        self.to(device)

    def forward(self, state, new_state, action):
        inverse = F.elu(self.inverse(T.cat([state, new_state], dim=1)))
        pi_logits = self.pi_logits(inverse)

        # from [T] to [T,1]
        action = action.reshape((action.size()[0], 1))
        forward_input = T.cat([state, action], dim=1)
        dense = F.elu(self.dense1(forward_input))
        state_ = self.new_state(dense)

        return pi_logits, state_

    def calc_loss(self, state, new_state, action):
        state = T.tensor(np.array(state), dtype=T.float)
        action = T.tensor(np.array(action), dtype=T.float)
        new_state = T.tensor(np.array(new_state), dtype=T.float)

        pi_logits, state_ = self.forward(state, new_state, action)

        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1-self.beta)*inverse_loss(pi_logits, action.to(T.long))

        forward_loss = nn.MSELoss()
        L_F = self.beta*forward_loss(state_, new_state)

        intrinsic_reward = self.alpha*((state_ - new_state).pow(2)).mean(dim=1)
        return intrinsic_reward, L_I, L_F
