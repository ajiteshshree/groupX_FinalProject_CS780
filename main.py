import torch.multiprocessing as mp
from parallel_env import ParallelEnv
import gymnasium as gym

if __name__ == "__main__":
    mp.set_start_method('spawn')

    env = gym.make("BipedalWalker-v3", hardcore=True)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] 
    n_episodes = 1000

    ParallelEnv(obs_size, action_size, n_episodes, env)
