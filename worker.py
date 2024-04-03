from heavy_lifting import *
import numpy as np
# import gymnasium as gym

def worker(n_episodes, optimizer, icm_optimizer, env):
    
    # env = gym.make("BipedalWalker-v3", hardcore=True)
    heavy_lifting = heavyLifting(env)

    for e in range(n_episodes):
        env.reset()
        story_memory = heavy_lifting.storeMemory()

        lossFn = heavy_lifting.lossFn()
        optimizer.zero_grad()
        icm_optimizer.zero_grad()
        lossFn.backward()
        optimizer.step()
        icm_optimizer.step()



