# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting
import time
from collections import deque


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Actor head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Critic head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))
        # Actor head
        probs = F.softmax(self.layers[-2](x), dim=-1)
        # Critic head
        value = self.layers[-1](x)

        return torch.squeeze(probs, -1), torch.squeeze(value, -1)


class A2C(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], env.action_space.n, self.options.layers
        )
        self.policy = self.create_greedy_policy()

        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.options.alpha)

        self.start_time = time.time()
        self.total_training_time = 0
        self.convergence_check_episodes = 50
        self.convergence_threshold = 300
        self.rewards_history = deque(maxlen=self.convergence_check_episodes)
        self.converged = False
        self.previous_avg = 0

    def moving_average(self, data, window_size):
        data_list = list(data)
        if len(data_list) < window_size:
            return np.mean(data_list)
        return np.mean(data_list[-window_size:])

    def has_converged(self):
        if len(self.rewards_history) < self.convergence_check_episodes:
            self.consecutive_convergence_count = 0
            return False

        current_avg = self.moving_average(self.rewards_history, self.convergence_check_episodes)
        if np.abs(current_avg - self.previous_avg) < self.convergence_threshold:
            self.consecutive_convergence_count += 1
        else:
            self.consecutive_convergence_count = 0
        print(f"current_avg {current_avg}, previous_avg {self.previous_avg}, consecutive_convergence_count {self.consecutive_convergence_count}")
        # return self.consecutive_convergence_count >= self.convergence_check_episodes


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

    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The critic's value estimate (as a tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, value = self.actor_critic(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value

    def update_actor_critic(self, advantage, prob, value):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            prob: Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        actor_loss = self.actor_loss(advantage.detach(), prob).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()

        loss = actor_loss + critic_loss

        # Update actor critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Perform an action in the env.
            self.options.gamma: Gamma discount factor.
            self.actor_critic(state): Returns the action probabilities and
                the critic's estimate at a given state.
            torch.as_tensor(state, dtype=torch.float32): Converts a numpy array
                'state' to a tensor.
            self.update_actor_critic(advantage, prob, value): Update actor critic. 
        """

        state, _ = self.env.reset()
        total_reward = 0
        self.previous_avg = self.moving_average(self.rewards_history, self.convergence_check_episodes)

        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            # Run update_actor_critic()    #
            # only ONCE at EACH step in    #
            # an episode.                  # 
            ################################
            action, prob, value = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            advantage = torch.tensor(reward + (1 - int(done)) * self.options.gamma * self.actor_critic(torch.tensor(next_state, dtype=torch.float32))[1] - value, dtype=torch.float32)
            self.update_actor_critic(advantage, prob, value)
            state = next_state
            if done:
                break

        self.rewards_history.append(total_reward)
        if self.has_converged():
            end_time = time.time()
            self.total_training_time = end_time - self.start_time
            self.converged = True
            print(f"Algorithm converged in {self.total_training_time} seconds.")

    def actor_loss(self, advantage, prob):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient.

        args:
            advantage: Advantage of the chosen action.
            prob: Probability associated with the chosen action.

        Use:
            torch.log: Element-wise logarithm.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################)
        return -torch.log(prob) * advantage

    def critic_loss(self, advantage, value):
        """
        The integral of the critic gradient

        args:
            advantage: Advantage of the chosen action.
            value: Critic's state value estimate.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        return -value * advantage

    def __str__(self):
        return "A2C"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
