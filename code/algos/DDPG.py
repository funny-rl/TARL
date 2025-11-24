import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .replay_buffer.naive_buffer import NaiveReplayBuffer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))
        return a
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPG():
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr,
        discount_factor,
        buffer_size,
        batch_size,
        hidden_dim,
        device
    ):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.max_action: float = max_action
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        
        self.lr: float = lr
        self.discount_factor: float = discount_factor
        self.device: str = device
        self.buffer_size: int = buffer_size
        self.tau: float = 0.005
        self.initial_lr: float = self.lr
        self.final_lr: float = 0.1 * self.initial_lr
        
        self.Actor= Actor(state_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.target_Actor = deepcopy(self.Actor).to(self.device)
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.lr)
        
        self.Critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_Critic = deepcopy(self.Critic).to(self.device)
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.lr)
    
        self.replay_buffer = NaiveReplayBuffer(
            max_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
    def select_action(self, state, deterministic = False):
        with torch.no_grad():
            action = self.Actor(state).cpu()
        if not deterministic:
            action = action + torch.randn(self.action_dim) * (0.1 * self.max_action)
        return action.clip(-self.max_action, self.max_action).numpy().flatten()

    def lr_decay(
        self, 
        training_rate,
    ):
        cosine = 0.5 * (1 + np.cos(np.pi * training_rate))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine
        for param_group in self.Actor_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.Critic_optimizer.param_groups:
            param_group['lr'] = self.lr

    def select_repetition(
        self, 
        state, 
        action, 
        warmup = False, 
        deterministic = False
    ):
        return 1
    
    def add(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done,
        skip_states = None,
        skip_rewards = None
    ):
        self.replay_buffer.add(
            state,
            action, 
            reward,
            next_state, 
            done, 
        )
        
    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        target_Q = self.target_Critic(
            torch.cat(
                [
                    next_states, 
                    self.target_Actor(next_states) / self.max_action
                ], 
                dim=-1
            )
        )
        target_Q = rewards + (1 - dones) * self.discount_factor * target_Q.detach()

        current_Q = self.Critic(
            torch.cat(
                [
                    states, actions / self.max_action
                ], 
                dim=-1
            )
        ) 
        critic_loss = F.mse_loss(current_Q, target_Q)
        td_error = (current_Q - target_Q).mean().item()
        
        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()
        
        actor_loss = -self.Critic(torch.cat([states, self.Actor(states) / self.max_action], dim=-1)).mean()
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()
        
        for param, target_param in zip(self.Actor.parameters(), self.target_Actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.Critic.parameters(), self.target_Critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        log_dict = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "td_error": td_error
        }
        
        return log_dict