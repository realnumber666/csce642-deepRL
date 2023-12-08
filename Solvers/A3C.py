import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from multiprocessing import Process, Manager
from Solvers.Abstract_Solver import AbstractSolver, Statistics
from lib import plotting
from Solvers.A2C import ActorCriticNetwork

import gymnasium as gym
import optparse
import sys
import os
import random
import numpy as np
import torch

gym.logger.set_level(40)

if "../" not in sys.path:
    sys.path.append(
        "../../../../Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/3e653f5e4f42f7963b72080e662cabad/Message/MessageTemp/597b064d27c9c138ba01989eda016794/File/csce642-deepRL/")

from lib import plotting
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
import matplotlib
import matplotlib.pyplot as plt

class A3C(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], env.action_space.n, self.options.layers
        )
        self.global_optimizer = Adam(self.actor_critic.parameters(), lr=self.options.alpha)

        self.manager = Manager()
        self.statistics = self.manager.list([0, 0, 0])

        self.options.num_workers = 4
        self.options.num_episodes_per_worker = 5
        # self.options.num_episodes_per_worker = int(self.options.episodes / self.options.num_workers)
        
        self.workers = [A3CWorker(self.statistics, self.actor_critic, self.global_optimizer, i, env, options) for i in range(self.options.num_workers)]
        self.policy = self.create_greedy_policy()

    def train_episode(self):
        processes = []
        for worker in self.workers:
            p = Process(target=worker.work)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.actor_critic(state)[0]).detach().numpy()

        return policy_fn
    
    def __str__(self):
        return "A3C"
    
    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)


class A3CWorker:
    def __init__(self, global_statistics, global_actor_critic, global_optimizer, worker_id, env, options):
        self.global_actor_critic = global_actor_critic
        self.global_optimizer = global_optimizer
        self.global_statistics = global_statistics


        self.worker_id = worker_id  
        self.env = env
        self.options = options
        self.local_actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], env.action_space.n, options.layers
        )
        self.local_actor_critic.load_state_dict(global_actor_critic.state_dict())
        self.local_optimizer = Adam(self.local_actor_critic.parameters(), lr=options.alpha)
    

    def calc_reward(self, state):
        # Create a new reward function for the CartPole domain that takes into account the degree of the pole
        try:
            domain = self.env.unwrapped.spec.id
        except:
            domain = self.env.name
        if domain == "CartPole-v1":
            x, x_dot, theta, theta_dot = state
            r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
            r2 = (
                self.env.theta_threshold_radians - abs(theta)
            ) / self.env.theta_threshold_radians - 0.5
            return r1 + r2
        return 0

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward += self.calc_reward(next_state)
        # Update statistics

        if self.worker_id == 0:
            self.global_statistics[Statistics.Rewards.value] += reward
            self.global_statistics[Statistics.Steps.value] += 1 
            # print(f"global statistics {self.global_statistics}")
        return next_state, reward, terminated or truncated, info
    

    def work(self):
        for episode in range(self.options.num_episodes_per_worker):
            state, _ = self.env.reset()
            done = False

            # if self.worker_id == 0:
            #     self.global_statistics[Statistics.Episode.value] += 1

            for step in range(self.options.steps):
                action, prob, value = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                # print(f'step {step}')
                # print(f'episode {episode}')
                # print(f'num_episodes_per_worker {self.options.num_episodes_per_worker}')  
                # print(f"action {action} {type(action)}")
                # print(f"next_state {next_state} {type(next_state)}")
                # print(f"reward {reward} {type(reward)}")
                # print(f"value {value} {type(value)}")
                self.update_local_model(state, action, reward, next_state, done, prob, value)
                if done:
                    # print(f"global statistics {self.global_statistics}")
                    # if self.worker_id == 0:
                    #     self.stats.episode_rewards.append(self.global_statistics[Statistics.Rewards.value])
                    #     self.stats.episode_lengths.append(self.global_statistics[Statistics.Steps.value])
                    self.sync_with_global()
                    break
                state = next_state

    def update_local_model(self, state, action, reward, next_state, done, prob, value):
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        _, next_value = self.local_actor_critic(next_state_tensor)

        td_target = reward + self.options.gamma * next_value * (1 - int(done))
        advantage = td_target - value

        actor_loss = -torch.log(prob) * advantage
        critic_loss = F.smooth_l1_loss(value, td_target.detach())

        total_loss = actor_loss + critic_loss

        self.local_optimizer.zero_grad()
        total_loss.backward()
        self.local_optimizer.step()

    def sync_with_global(self):
        for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

        self.global_optimizer.step()

        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, value = self.local_actor_critic(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value

