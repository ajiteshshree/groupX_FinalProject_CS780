import gymnasium as gym
import torch as T
import numpy as np

def render_environment(env_id, agent):
    env = gym.make(env_id, render_mode = 'human')
    obs = env.reset()[0]
    obs = np.array(obs)
    done = False
    while not done:
        state = T.tensor(np.array([obs]), dtype=T.float)
        action, _, _, _ = agent.forward(state, hx=None)
        obs_, _, done, _, _ = env.step(action.item())
        obs_ = np.array(obs_)
        obs = obs_
        # env.render()
    # env.close()