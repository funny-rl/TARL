import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils

from copy import deepcopy

from ._modules import QNet, DuelDQN
from .buffer.naive_buffer import NaiveReplayBuffer



class DQN:
    def __init__(
        self,
        state_dim,
        action_dim, 
        n_actions,
        lr,
        gamma,
        buffer_size,
        batch_size,
        hidden_dim,
        e_greedy_type,
        e_decay,
        max_epsilon,
        min_epsilon,
        use_image,
        use_lr_decay,
        use_hard_update,
        update_interval,
        tau,
        use_dueling,
        device
    ):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.n_actions: int = n_actions
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        self.update_interval: int = update_interval
        self.e_decay: int = e_decay
        
        self.lr: float = lr
        self.initial_lr: float = lr
        self.final_lr: float = lr * 0.1 
        
        self.tau: float = tau
        self.gamma: float = gamma
        self.epsilon: float = max_epsilon
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        
        self.e_greedy_type: str = e_greedy_type
        self.use_image: bool = use_image
        self.device: str = device
        
        self.use_dueling: bool = use_dueling
        self.use_lr_decay: bool = use_lr_decay
        self.use_hard_update: bool = use_hard_update
        
        if self.use_dueling:
            self.Actor = DuelDQN(
                self.state_dim,
                self.n_actions,
                self.hidden_dim,
                self.use_image,
            ).to(self.device)
        else:
            self.Actor = QNet(
                self.state_dim, 
                self.n_actions, 
                self.hidden_dim,
                self.use_image,
            ).to(self.device)
        
        self.Actor_optimizer = optim.Adam(
            self.Actor.parameters(), 
            lr=self.lr,
        )
        self.target_Actor = deepcopy(self.Actor).to(self.device)
        self.loss_func = nn.SmoothL1Loss()

        self.replay_buffer = NaiveReplayBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
    def epsilon_decay(self, training_steps):
        training_steps = torch.tensor(training_steps, dtype=torch.float32)
        if self.e_greedy_type == "linear":
            self.epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * torch.clamp(training_steps / self.e_decay, 0.0, 1.0).item()
        elif self.e_greedy_type == "exponential":
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * torch.exp(-1.0 * training_steps / self.e_decay).item()
        else:
            raise NotImplementedError(f"Epsilon greedy type {self.e_greedy_type} is not supported.")

    def lr_decay(self, training_rate):
        cosine = 0.5 * (1 + torch.cos(torch.pi * torch.tensor(training_rate)))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine.item()
        for param_group in self.Actor_optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def select_action(
        self,
        state: torch.Tensor, 
        deterministic: bool = False,
    ) -> int:
        if deterministic or torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.Actor(state)
                action = q_values.argmax(dim=-1) 
        else:
            action = torch.randint(0, self.n_actions, (1,))
        return action.cpu().numpy()
    
    def select_repetition(
        self, 
        state,
        action = None,
        deterministic = False
    ) -> int:
        if deterministic:
            return 1, None
        else:
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
            done
        )

    def update(self, training_steps: int) -> dict:

        (
            states, 
            actions, 
            rewards, 
            next_states, 
            not_dones,
        ) = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_q_values = torch.max(self.target_Actor(next_states), dim=-1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * not_dones * next_q_values
        
        q_values = self.Actor(states).gather(-1, index=actions.long())
        q_loss = self.loss_func(q_values, target_q_values)
        
        self.Actor_optimizer.zero_grad()
        q_loss.backward()
        self.Actor_optimizer.step()
        
        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_Actor.load_state_dict(self.Actor.state_dict())
        else:
            for target_param, param in zip(self.target_Actor.parameters(), self.Actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        log_dict: dict = {
            "q_loss": q_loss.clone().cpu().item(),
            "td_error": (target_q_values - q_values).clone().cpu().mean().item(),
        }
        
        return log_dict

    def save_model(self, path: str):
        torch.save(self.Actor.state_dict(), path + "actor.pth")