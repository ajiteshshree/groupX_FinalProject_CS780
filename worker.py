from ICM import *
from mainNetwork import *
import gymnasium as gym
import numpy as np
import torch

class worker():
    def __init__(self, env):
        
        self.env = env
        self.memory = []
        self.icm_worker = ICM(env.observation_space.shape[0],env.action_space.shape[0])
        self.boomer_worker = mainLoop(env.observation_space.shape[0],env.action_space.shape[0])

    def storeMemory(self):

        # observation, reward, terminated, truncated, info = self.env.step(action)
            
        info = self.env.reset()
        current_state = info[0]
        done = False
        while done == False:
            action = self.boomer_worker.forwardPass(current_state)
            next_state, extrinsic_reward, done, truncated, info = self.env.step(action)
            pred_nextState = self.icm_worker.forwardModel(current_state, action)
            loss_forward, intrinsic_reward = self.icm_worker.forwardLossAndReward(next_state)

            current_state = np.array(current_state)
            next_state = np.array(next_state)

            self.memory.append([current_state, action, next_state, \
                                pred_nextState, extrinsic_reward, intrinsic_reward, loss_forward])
            
            if truncated == True:
                self.memory = []
                info = self.env.reset()
                current_state = info[0]
                done = False

        return self.memory



