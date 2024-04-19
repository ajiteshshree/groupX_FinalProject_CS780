import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
from render_environment import render_environment

os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'CartPole-v1'
    # env_id = 'FrozenLake-v1'
    n_threads = 12
    n_actions = 2
    # n_actions = 4
    input_shape = [4]
    # input_shape = [1]
    env = ParallelEnv(env_id=env_id, n_threads=n_threads,
                      n_actions=n_actions, input_shape=input_shape, icm=True)
    # env.run_training()
    render_environment(env_id, env.global_actor_critic)
