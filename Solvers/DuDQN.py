# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import random
import time
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class DuDQN(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
                str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # Create Q-network
        self.model = QFunction(
            env.observation_space.shape[0],
            env.action_space.n,
            self.options.layers,
        )

        # Create behavior model
        self.behavior_model = QFunction(
            env.observation_space.shape[0],
            env.action_space.n,
            self.options.layers,
        )

        # Create target Q-network
        self.target_model = deepcopy(self.model)
        # Set up the optimizer
        self.optimizer = AdamW(
            self.behavior_model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # Number of training steps so far
        self.n_steps = 0

        # 收敛时间统计逻辑
        self.start_time = time.time()
        self.total_training_time = 0  # 总训练时间
        self.convergence_check_episodes = 50  # 用于检测收敛的连续周期数
        self.convergence_threshold = 300  # 判断收敛的奖励标准差阈值
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

    def update_target_model(self):
        self.target_model.load_state_dict(self.behavior_model.state_dict())

    def epsilon_greedy(self, state):
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            The probabilities (as a Numpy array) associated with each action for 'state'.

        Use:
            self.env.action_space.n: Number of avilable actions
            self.torch.as_tensor(state): Convert Numpy array ('state') to a tensor
            self.model(state): Returns the predicted Q values at a
                'state' as a tensor. One value per action.
            torch.argmax(values): Returns the index corresponding to the highest value in
                'values' (a tensor)
        """
        # Don't forget to convert the states to torch tensors to pass them through the network.
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        nA = self.env.action_space.n
        action_prob = np.ones(nA, dtype=float) * self.options.epsilon / nA
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.behavior_model(state_tensor)
        best_action = torch.argmax(q_values, dim=1).item()
        action_prob[best_action] += (1. - self.options.epsilon)
        return action_prob

    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Returns:
            The target q value (as a tensor) of shape [len(next_states)]
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32)
        next_q_values = self.target_model(next_states_tensor).max(1)[0].detach()
        targets = rewards + self.options.gamma * next_q_values * (1 - dones)
        return targets

    def replay(self):
        """
        TD learning for q values on past transitions.

        Use:
            self.target_model(state): predicted q values as an array with entry
                per action
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Use behavior model to select action
            behavior_q_values = self.behavior_model(next_states)
            max_behavior_actions = torch.argmax(behavior_q_values, dim=1)

            # Use target model to select q
            target_q_values = self.target_model(next_states).detach()
            target_q = target_q_values.gather(1, max_behavior_actions.unsqueeze(1)).squeeze(-1)

            td_target = rewards + self.options.gamma * target_q * (1 - dones)

            # Current Q-values
            current_q = self.behavior_model(states)
            # Q-values for actions in the replay memory
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            # Calculate loss
            loss = self.loss_fn(current_q, td_target)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.behavior_model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation. Finds the optimal greedy policy
        while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return probabilities of actions.
            np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
                probabilites 'prob'.
            self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps
        """

        # Reset the environment
        state, _ = self.env.reset()
        total_reward = 0
        self.previous_avg = self.moving_average(self.rewards_history, self.convergence_check_episodes)

        for t in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action_prob = self.epsilon_greedy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done, _ = self.step(action)
            self.memorize(state, action, reward, next_state, done)
            self.replay()

            if t % self.options.update_target_estimator_every == 0:
                self.update_target_model()

            state = next_state
            total_reward += reward
            if done:
                break

        # 更新奖励历史并检查是否收敛
        self.rewards_history.append(total_reward)
        if self.has_converged():
            end_time = time.time()
            self.total_training_time = end_time - self.start_time
            self.converged = True
            print(f"Algorithm converged in {self.total_training_time} seconds.")

    def __str__(self):
        return "DuDQN"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, 25, final=final)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn
