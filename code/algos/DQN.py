import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from .replay_buffer.naive_buffer import NaiveReplayBuffer

class QNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim, data_type):
        super(QNet, self).__init__()
        self.data_type = data_type
        
        if self.data_type == "pixels":
            C_dim = state_dim[0] # 3
            h_dim = state_dim[1]
            w_dim = state_dim[2] 
            self.cnn = nn.Sequential(
                nn.Conv2d(C_dim, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            self.test_tensor = torch.zeros((1, C_dim, h_dim, w_dim))
            with torch.no_grad():
                cnn_output_dim = self.cnn(self.test_tensor).shape[1]
            
            self.fc = nn.Sequential(
                nn.Linear(cnn_output_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
                
            )
        elif self.data_type == "features":
            self.fc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
        else:
            raise ValueError(f"data_type {self.data_type} is not supported.")

    def forward(self, x):
        if self.data_type == "pixels":
            x = self.cnn(x)
            x = self.fc(x)
        elif self.data_type == "features":
            x = self.fc(x)
        return x

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim, 
        n_actions,
        lr,
        discount_factor,
        buffer_size,
        batch_size,
        hidden_dim,
        e_greedy_type,
        power_value,
        min_epsilon,
        data_type,
        device
    ):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.n_actions: int = n_actions
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        self.buffer_size: int = buffer_size
        
        self.device: str = device
        self.e_greedy_type: str = e_greedy_type
        
        self.lr: float = lr
        self.tau: float = 0.005
        self.epsilon: float = 1.0 
        self.min_epsilon: float = min_epsilon
        self.power_value: float = power_value
        self.discount_factor: float = discount_factor
        self.initial_lr: float = self.lr
        self.final_lr: float = 0.1 * self.initial_lr
        
        self.Actor = QNet(state_dim, self.n_actions, self.hidden_dim, data_type).to(self.device)
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=lr)
        self.target_Actor = deepcopy(self.Actor).to(self.device)
        
        self.replay_buffer = NaiveReplayBuffer(
            max_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
    def epsilon_decay(self, training_rate):    
        if self.e_greedy_type == "linear":
            self.epsilon = min(1.0, max(self.min_epsilon, 1.0 - training_rate))
        elif self.e_greedy_type == "power":
            self.epsilon = (min(1.0, max(self.min_epsilon, (1.0 - training_rate) ** self.power_value)))
        else:
            raise ValueError(f"e_greedy_type {self.e_greedy_type} is not supported.")

    def lr_decay(
        self, 
        training_rate,
    ):
        training_rate = min(1.0, max(0.0, training_rate))
        cosine = 0.5 * (1 + np.cos(np.pi * training_rate))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine
        for param_group in self.Actor_optimizer.param_groups:
            param_group['lr'] = self.lr

    def select_action(self, state, deterministic = False):
        if deterministic or torch.rand(1).item() >= self.epsilon:
            with torch.no_grad():
                q_values = self.Actor(state)
            action = q_values.argmax(dim=-1).unsqueeze(0)
        else:
            action = torch.randint(0, self.n_actions, (1,)).unsqueeze(0)

        if action.shape[-1] == 1:
            action = action.item()
        elif action.shape[-1] > 1:
            action = action.numpy().flatten()
        return action
    
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
        
        current_q_values = self.Actor(states).gather(1, actions.long())
        
        with torch.no_grad():
            next_q_values = torch.max(self.target_Actor(next_states), dim = -1, keepdim = True)[0]
            target_q_values = rewards + self.discount_factor * (1 - dones) * next_q_values
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.Actor_optimizer.zero_grad()
        loss.backward()
        self.Actor_optimizer.step()
        
        td_error = (target_q_values - current_q_values).mean().item()
        
        # Soft update of target network
        for target_param, param in zip(self.target_Actor.parameters(), self.Actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        return {"td_error": td_error}
