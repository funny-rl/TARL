import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from ._modules import Continuous_Q_Actor, Continuous_Q_Critic
from .buffer.naive_buffer import NaiveReplayBuffer

class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr,
        gamma,
        buffer_size,
        batch_size,
        hidden_dim,
        use_image,
        use_lr_decay,
        use_hard_update,
        update_interval,
        tau,
        expl_noise,
        device
    ):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.max_action: float = max_action
        
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        self.update_interval: int = update_interval
        
        self.lr: float = lr
        self.initial_lr: float = lr
        self.final_lr: float = lr * 0.1 
        self.expl_noise: float = expl_noise
        
        self.tau: float = tau
        self.gamma: float = gamma
        
        self.device: str = device
        
        self.use_image: bool = use_image
        self.use_lr_decay: bool = use_lr_decay
        self.use_hard_update: bool = use_hard_update
        
        self.Actor = Continuous_Q_Actor(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_action,
        ).to(self.device)

        self.Critic = Continuous_Q_Critic(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
        ).to(self.device)

        self.Actor_optimizer = torch.optim.Adam(
            self.Actor.parameters(),
            lr=self.lr,
        )
        self.Critic_optimizer = torch.optim.Adam(
            self.Critic.parameters(),
            lr=self.lr,
        )

        self.target_Actor = deepcopy(self.Actor).to(self.device)
        self.target_Critic = deepcopy(self.Critic).to(self.device)
        
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = NaiveReplayBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )

    def lr_decay(self, training_rate):
        cosine = 0.5 * (1 + torch.cos(torch.pi * torch.tensor(training_rate)))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine.item()
        for param_group in self.Actor_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.Critic_optimizer.param_groups:
            param_group['lr'] = self.lr

    def select_action(
        self, 
        state,
        deterministic = False
    ) -> torch.Tensor:
        with torch.no_grad():
            action = self.Actor(state).flatten()
        if not deterministic:
            action += torch.normal(0, self.expl_noise, size=action.shape).to(self.device)
        action = torch.clamp(action, -self.max_action, self.max_action)
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
    
    def update(self, training_steps):

        (
            states, 
            actions, 
            rewards, 
            next_states, 
            not_dones,
        ) = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.target_Actor(next_states)
            target_Q_values = self.target_Critic(
                torch.cat([next_states, next_actions], dim=-1)
            )
            target_Q = rewards + self.gamma * not_dones * target_Q_values
        
        current_Q = self.Critic(
            torch.cat([states, actions], dim=-1)
        )
        critic_loss = self.loss_fn(current_Q, target_Q.detach())

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()

        curr_actions = self.Actor(states)
        actor_loss = -self.Critic(
            torch.cat([states, curr_actions], dim=-1)
        ).mean()
        
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()

        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_Actor.load_state_dict(self.Actor.state_dict())
                self.target_Critic.load_state_dict(self.Critic.state_dict())
        else:
            for param, target_param in zip(self.Critic.parameters(), self.target_Critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.Actor.parameters(), self.target_Actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        log_dict = {
            "critic_loss": critic_loss.item(),
            "td_error": (target_Q - current_Q).clone().cpu().mean().item(),
            "actor_loss": actor_loss.item(),
        }
        return log_dict