import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from Solvers.Abstract_Solver import AbstractSolver
from Solvers.A2C import ActorCriticNetwork
from lib import plotting

class PPO_Clip(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], env.action_space.n, options.layers
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=options.alpha)

        # PPO Hyperparameters
        # self.eps_clip = options.eps_clip
        # self.K_epochs = options.K_epochs
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.policy = self.create_greedy_policy()
        

    def create_greedy_policy(self):
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.actor_critic(state)[0]).detach().numpy()
        return policy_fn

    def select_action(self, state, memory):
        state = torch.as_tensor(state, dtype=torch.float32)
        action_probs, state_value = self.actor_critic(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        memory.state_values.append(state_value)

        return action.item()

    def evaluate(self, state, action):
        action_probs, state_value = self.actor_critic(state)
        # print("============")
        # print(f"state {state} {type(state)}")
        # print(f"action_probs {action_probs} {type(action_probs)}")
        # print(f"state_value {state_value} {type(state_value)}")
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def train_episode(self):
        memory = Memory()
        state, _ = self.env.reset()

        for _ in range(self.options.steps):
            action = self.select_action(state, memory)
            next_state, reward, done, _ = self.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if done:
                break
            state = next_state

        # Update policy
        self.update_policy(memory)

    def compute_gae(self, next_value, rewards, state_values, masks, gamma=0.99, tau=0.95):
        gae = 0
        returns = []
        # print(f"reward {rewards} {type(rewards)}")
        # print(f"state_values {state_values} {type(state_values)}")
        # print(f"masks {masks} {type(masks)}")
        # print(f"next_value {next_value} {type(next_value)}")

        # next_value = next_value.expand_as(state_values[-1])
        state_values = torch.cat((state_values,next_value.reshape(1)), dim=0)
        
        # print(f"new_state_values {state_values} {type(state_values)}")
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * state_values[step + 1] * masks[step] - state_values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + state_values[step])
        return returns


    def update_policy(self, memory):
        # Convert list to tensor
        rewards = torch.tensor(memory.rewards)
        masks = torch.tensor(memory.is_terminals).float()
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        # Calculate state values and next state values
        with torch.no_grad():
            state_values = torch.stack(memory.state_values).squeeze(-1)
            next_state_value = self.actor_critic(old_states[-1])[1]
            # next_state_values = torch.cat((state_values[1:], torch.tensor([0.0])))

        # Compute returns and advantages
        returns = self.compute_gae(next_state_value, rewards, state_values, masks, self.options.gamma)
        returns = torch.tensor(returns).detach()
        advantages = returns - state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_values, returns)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def __str__(self):
        return "PPO_Clip"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)



class PPO_Penalty(PPO_Clip):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # self.kl_coeff = options.kl_coeff  # KL 系数
        self.kl_coeff = 0.01


    def update_policy(self, memory):
        rewards = torch.tensor(memory.rewards)
        masks = torch.tensor(memory.is_terminals).float()
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        # Calculate state values and next state values
        with torch.no_grad():
            state_values = torch.stack(memory.state_values).squeeze(-1)
            next_state_value = self.actor_critic(old_states[-1])[1]

        # Compute returns and advantages
        returns = self.compute_gae(next_state_value, rewards, state_values, masks, self.options.gamma)
        returns = torch.tensor(returns).detach()
        advantages = returns - state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # print(f"old_states {old_states} {type(old_states)}")
            # print(f"old_actions {old_actions} {type(old_actions)}")

            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            # print(f"logprobs {logprobs} {type(logprobs)}")
            # print(f"state_values {state_values} {type(state_values)}")
            # print(f"old_logprobs {old_logprobs+1e-4} {type(old_logprobs+1e-4)}")
            # print(f"logprobs {logprobs} {type(logprobs)}")
            # Calculate KL divergence (old_logprobs vs logprobs)
            kl_div = F.kl_div(old_logprobs, logprobs, reduction='batchmean', log_target=True)

            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # print(f"kl_div {kl_div} {type(kl_div)}")
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_values, returns) + self.kl_coeff * kl_div
            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_values, returns)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def __str__(self):
        return "PPO_Penalty"

# Helper classes and functions
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.state_values = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

# # Usage
# env = gym.make('YourEnvName')
# eval_env = gym.make('YourEvalEnvName')
# options = ... # Set your options
# ppo_solver = PPO(env, eval_env, options)
# ppo_solver.train_episode()
