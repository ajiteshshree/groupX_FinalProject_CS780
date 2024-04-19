import torch.multiprocessing as mp
from ICM import *
from mainNetwork import *
from worker import worker
from sharedAdam import *

class ParallelEnv:
    def __init__(self, obs_size, action_size, n_episodes, env,  n_threads = 5):
        
        names = [str(i) for i in range(1, n_threads+1)]

        global_main_loop = mainLoop(obs_size, action_size)
        global_main_loop.share_memory()
        global_optim = SharedAdam(global_main_loop.parameters())

        global_icm = ICM(obs_size, action_size)
        global_icm.share_memory()
        global_icm_optim = SharedAdam(global_icm.parameters())

        self.ps = [mp.Process(target=worker,
                                args=(n_episodes, global_optim, global_icm_optim, env))
                    for name in names]
        
        [p.start() for p in self.ps]
        [p.join() for p in self.ps]