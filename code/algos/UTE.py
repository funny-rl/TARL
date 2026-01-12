import torch
import random
import collections

import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils.utils import action_transform

from .buffer.repetition_buffer import SkipBuffer
from ._modules import Ensemble_DQN

class UCB:
    """
    Determine the index of the arms in terms of solving a multi-armed bandit problem
    Attributes:
      data           : list that stores the index and average reward of the arms
      num_arms  (int): number of arms used in multi-armed bandit problem
      epsilon (float): probability to select the index of the arms used in multi-armed bandit problem
      beta    (float): weight between frequency and mean reward
      count     (int): if count is less than num_arms, index is count because of trying to pick every arm at least once
    """

    def __init__(self, num_arms, window_size, epsilon, beta):
        """
        num_arms    (int): number of arms used in multi-armed bandit problem
        window_size (int): size of window used in multi-armed bandit problem
        epsilon   (float): probability to select the index of the arms used in multi-armed bandit problem
        beta      (float): weight between frequency and mean reward
        """
        
        self.data = collections.deque(maxlen=window_size)
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.beta = beta
        self.count = 0

    def pull_index(self):
        """
        pull index to determine value of betas and gammas
        Returns:
          index (float): index of arms 
        """
        
        if self.count < self.num_arms:
            index = self.count
            self.count += 1
            
        else:
            if random.random() > self.epsilon:
                N = np.zeros(self.num_arms)
                mu = np.zeros(self.num_arms)
                
                for j, reward in self.data:
                    N[j] += 1
                    mu[j] += reward
                mu = mu / (N + 1e-10)
                index = np.argmax(mu + self.beta * np.sqrt(1 / (N + 1e-6)))
                
            else:
                index = np.random.choice(self.num_arms)
        return index

    def push_data(self, datas):
        """
        push datas to UCB's data list
        Args:
          datas :store index of arms and resulting reward         
        """
        
        self.data += [(j, reward) for j, reward in datas]

class UTE:
    def __init__(
        self,
        base_agent,
        rep_batch_size,
        rep_buffer_size,
        max_repetition,
        e_greedy_type,
        e_decay,
        max_epsilon,
        min_epsilon,
        num_ensemble,
        use_adaptive_uncertainty,
        uncertainty_factor,
        use_act_skip_buf,
        prev_buffer_save
    ):
        self.base_agent = base_agent
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim
        
        self.is_continuous = False if hasattr(self.base_agent, "n_actions") else True
        
        if self.is_continuous:
            self.n_actions = None
        else:
            self.n_actions = base_agent.n_actions
        
        self.rep_buffer_size: int = rep_buffer_size
        self.rep_batch_size: int = rep_batch_size
        self.hidden_dim: int = base_agent.hidden_dim
        self.update_interval: int = base_agent.update_interval
        self.e_decay: int = e_decay
        self.max_repetition: int = max_repetition
        self.num_ensemble: int = num_ensemble
        
        self.lr: float = base_agent.lr
        self.initial_lr: float = base_agent.initial_lr
        self.final_lr: float = base_agent.final_lr
        self.tau: float = base_agent.tau
        self.gamma: float = base_agent.gamma
        self.epsilon: float = max_epsilon
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        self.use_adaptive_uncertainty: bool = use_adaptive_uncertainty
        if self.use_adaptive_uncertainty:
            self.lambdas = [-1.5, -1.0, -0.5, -0.2, 0.0, +0.2, +0.5, +1.0, +1.5]
            num_arms = len(self.lambdas)
            self.ucb = UCB(
                num_arms = num_arms, 
                window_size = 500, 
                epsilon = 0.1, 
                beta = 0.5
            )
        else:
            self.uncertainty_factor: float = uncertainty_factor

            
        self.bernoulli_probability: float = 0.5
        self.e_greedy_type: str = e_greedy_type
        self.device: str = base_agent.device
        
        self.use_image: bool = base_agent.use_image
        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.use_hard_update: bool = base_agent.use_hard_update
        self.use_act_skip_buf: bool = use_act_skip_buf

        self.Rep_Actor = Ensemble_DQN(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_repetition,
            self.use_image,
            self.num_ensemble,
            self.n_actions
        ).to(self.device)
        
        self.loss_fn = nn.SmoothL1Loss()
        self.Rep_Actor_optimizer = optim.Adam(self.Rep_Actor.parameters(), lr=self.lr)
        
        self.rep_replay_buffer = SkipBuffer(
            buffer_size=self.rep_buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            prev_buffer_save=prev_buffer_save,
            device=self.device
        )
        
    def epsilon_decay(self, training_steps):
        if hasattr(self.base_agent, "epsilon_decay"):
            self.base_agent.epsilon_decay(training_steps)
            self.epsilon = self.base_agent.epsilon
        else: 
            training_steps = torch.tensor(training_steps, dtype=torch.float32)
            if self.e_greedy_type == "linear":
                self.epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * torch.clamp(training_steps / self.e_decay, 0.0, 1.0).item()
            elif self.e_greedy_type == "exponential":
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * torch.exp(-1.0 * training_steps / self.e_decay).item()
            else:
                raise NotImplementedError(f"Epsilon greedy type {self.e_greedy_type} is not supported.")
            
    def adaptive_lambda(self):
        self.j = self.ucb.pull_index()
        self.uncertainty_factor = self.lambdas[self.j]
        self.ucb_datas: list = []
        
    def lr_decay(self, training_rate):
        self.base_agent.lr_decay(training_rate)
        self.lr = self.base_agent.lr
        for param_group in self.Rep_Actor_optimizer.param_groups:
            param_group['lr'] = self.lr
            
    def select_action(
        self,
        state,
        deterministic: bool = False
    ):
        return self.base_agent.select_action(state, deterministic)
    
    def select_repetition(
        self,
        state,
        action = None,
        deterministic: bool = False
    ):
        if (deterministic or torch.rand(1).item() > self.epsilon) and action is not None:
            with torch.no_grad():
                repetition_q_values = self.Rep_Actor(state, action)
                
                mean_q_values = torch.mean(repetition_q_values, dim=0)
                std_q_values = torch.std(repetition_q_values, dim=0)

                rep_Qs = mean_q_values + self.uncertainty_factor * std_q_values
                    
                repetitions = torch.argmax(rep_Qs, dim=-1).squeeze().item() + 1
            if deterministic:
                return repetitions, rep_Qs
        else:
            repetitions = torch.randint(1, self.max_repetition + 1, (1,)).item()

        return repetitions

    def add(
        self,
        skip_states,
        action,
        skip_rewards,
        skip_dones,
        next_skip_states,
        repetition
    ):
        self.rep_replay_buffer = self.rep_replay_buffer.transform(
            skip_states = skip_states,
            action = action,
            skip_rewards = skip_rewards,
            skip_dones = skip_dones,
            next_skip_states = next_skip_states,
            repetition = repetition,
            buffer = self.rep_replay_buffer
        )
        if self.use_act_skip_buf:
            self.base_agent.add_skip_buffer(
                buffer = self.rep_replay_buffer
            )
        
    def update(self, training_steps: int) -> dict:
        log_dict = self.base_agent.update(training_steps)
        
        (
            states, 
            actions, 
            reps, 
            rewards, 
            next_states, 
            not_dones
        ) = self.rep_replay_buffer.sample(self.rep_batch_size)
        
        rep_idx = reps.long() - 1
        
        with torch.no_grad():
            if self.is_continuous == True:
                next_actions = self.base_agent.target_Actor(next_states)
                next_q_values = self.base_agent.target_Critic(
                    torch.cat([next_states, next_actions], dim=-1)
                )
            else:
                next_actions = self.base_agent.target_Actor(next_states)
                next_q_values = torch.max(next_actions, dim=-1, keepdim=True)[0]
                
            target_Q = rewards + not_dones * (self.gamma ** reps) * next_q_values


        actions = action_transform(actions, self.n_actions, self.device)
        q_values = self.Rep_Actor(states, actions)
        
        masks = torch.bernoulli(torch.zeros((self.rep_batch_size, self.num_ensemble), device=self.device) + self.bernoulli_probability)
        
        cnt_losses = 0.0
        for k in range(self.num_ensemble):
            num_update = masks[:, k].sum()
            if num_update > 0:
                current_Q = q_values[k].gather(-1, index=rep_idx)
                loss = self.loss_fn(current_Q * masks[:, k], target_Q * masks[:, k]) / num_update
                cnt_losses += loss
        
        skip_losses = cnt_losses / self.num_ensemble
        
        self.Rep_Actor_optimizer.zero_grad()
        skip_losses.backward()
        self.Rep_Actor_optimizer.step()

        log_dict.update({
            "rep_q_loss": skip_losses.clone().cpu().item(),
            "rep_td_error": (q_values.mean(dim = 0) - target_Q).mean().clone().cpu().item(),
        })
        return log_dict
    
    def save_model(self, path: str):
        torch.save(self.Rep_Actor.state_dict(), path + "rep_actor.pth")
        self.base_agent.save_model(path)