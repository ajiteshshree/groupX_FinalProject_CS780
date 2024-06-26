import gymnasium as gym
import numpy as np
import torch as T
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
from utils import plot_learning_curve

def worker(name, input_shape, n_actions, global_agent, global_icm,
           optimizer, icm_optimizer, env_id, n_threads, icm=False):
    T_MAX = 20

    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape, n_actions)
        algo = 'ICM'
    else:
        intrinsic_reward = T.zeros(1)
        algo = 'A3C'

    memory = Memory()

    env = gym.make(env_id)

    t_steps, max_eps, episode, scores, avg_score = 0, 1000, 0, [], 0

    while episode < max_eps:

        obs = env.reset()[0]
        #since Frozen Lake env gives a number as obs instead of a list
        # we change it to a list here
        obs = np.array([obs])
        hx = T.zeros(1, 256)
        score, done, ep_steps = 0, False, 0
        while not done:
            state = T.tensor(np.array([obs]), dtype=T.float)
            action, value, log_prob, hx = local_agent.forward(state, hx)
            obs_, reward, done, truncated, info = env.step(action)

            #since Frozen Lake env gives a number as obs instead of a list
            # we change it to a list here
            obs_ = np.array([obs_])

            t_steps += 1
            ep_steps += 1
            score += reward

            # since sparse reward in Frozen lake, we need extrinsic reward to learn as well
            # reward = 0  # turn off extrinsic rewards
            memory.remember(obs, action, reward, obs_, value, log_prob)
            obs = obs_
            if ep_steps % T_MAX == 0 or done:
                states, actions, rewards, new_states, values, log_probs = \
                        memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = \
                            local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                             log_probs, intrinsic_reward)

                optimizer.zero_grad()
                hx = hx.detach()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()

                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                                        local_agent.parameters(),
                                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                                            local_icm.parameters(),
                                            global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())
                memory.clear_memory()

        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('{} episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'intrinsic_reward {:.2f} avg score (100) {:.1f}'.format(
                      algo, episode, name, n_threads,
                      t_steps/1e6, score,
                      T.sum(intrinsic_reward),
                      avg_score))
        episode += 1
    if name == '1':
        x = [z for z in range(episode)]
        fname = algo + '_FrozenLake_no_rewards.png'
        plot_learning_curve(x, scores, fname)
