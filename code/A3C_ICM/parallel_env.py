import torch.multiprocessing as mp
from actor_critic import ActorCritic
from icm import ICM
from shared_adam import SharedAdam
from worker import worker
from render_environment import render_environment

class ParallelEnv:
    def __init__(self, env_id, input_shape, n_actions, icm, n_threads=8):
        names = [str(i) for i in range(1, n_threads+1)]

        self.global_actor_critic = ActorCritic(input_shape, n_actions)
        self.global_actor_critic.share_memory()
        self.global_optim = SharedAdam(self.global_actor_critic.parameters())
        self.env_id = env_id
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.n_threads = n_threads
        self.icm = icm
        

        if not icm:
            self.global_icm = None
            self.global_icm_optim = None
        else:
            self.global_icm = ICM(input_shape, n_actions)
            self.global_icm.share_memory()
            self.global_icm_optim = SharedAdam(self.global_icm.parameters())

        self.ps = [mp.Process(target=worker,
                              args=(name, self.input_shape, self.n_actions,
                                    self.global_actor_critic, self.global_icm,
                                    self.global_optim, self.global_icm_optim, self.env_id,
                                    n_threads, icm))
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]


    # def all_threads_finished(self):
    #     for p in self.ps:
    #         if p.is_alive():
    #             return False
    #     return True

    # def render_environment(self):
    #     render_environment(self.env_id, self.global_actor_critic)

    # def run_optimized_global_algorithm(self):
    #     # Create new process to run the optimized global algorithm
    #     self.p_optimized = mp.Process(target=worker,
    #                              args=('optimized', self.input_shape, self.n_actions,
    #                                    self.global_actor_critic, self.global_icm,
    #                                    self.global_optim, self.global_icm_optim, self.env_id,
    #                                    self.n_threads, self.icm, True))
    #     # name, input_shape, n_actions, global_agent, global_icm,
    #     #    optimizer, icm_optimizer, env_id, n_threads, icm=False, run_optimized=False
        
    #     self.p_optimized.start()
    #     self.p_optimized.join()
       


    def run_training(self):
        if not any(p.is_alive() for p in self.ps):
            # [p.start() for p in self.p_optimized]
            # [p.join() for p in  self.p_optimized]
            if self.all_threads_finished():
                self.render_environment(env_id = self.env_id, agent = self.global_actor_critic)
                # self.run_optimized_global_algorithm()
        else:
            print("Some processes are already running. Cannot start training again.")
    
    

