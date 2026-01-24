import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from ._modules import Continuous_Q_Actor, Continuous_Q_Critic
from .buffer.TAAC_buffer import TAACBuffer
import torch.nn.functional as F

class TAAC:
    def __init__(
        self,
        base_agent,
        temperature,
        target_entropy_delta,
        seq_len,
    ):
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim
        self.max_action: float = base_agent.max_action

        self.lr: float = base_agent.lr
        self.gamma: float = base_agent.gamma

        self.buffer_size: int = base_agent.buffer_size
        self.batch_size: int = base_agent.batch_size
        self.hidden_dim: int = base_agent.hidden_dim
        self.use_image: bool = base_agent.use_image
        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.use_hard_update: bool = base_agent.use_hard_update
        self.update_interval: int = base_agent.update_interval

        self.initial_lr: float = base_agent.lr
        self.final_lr: float = base_agent.lr * 0.1 

        self.tau: float = base_agent.tau
        self.expl_noise: float = base_agent.expl_noise
        self.device: str = base_agent.device

        self.target_entropy_delta: float = target_entropy_delta
        self.target_entropy : float = -self.target_entropy_delta * np.log(self.target_entropy_delta) \
                                    - (1 - self.target_entropy_delta)*np.log(1 - self.target_entropy_delta)
        self.seq_len: int = seq_len
        
        
        
        
        self.temperature: float = temperature
        self.log_temperature = torch.tensor(np.log(temperature), dtype=torch.float32, device=self.device, requires_grad=True)
        

        
        self.Actor = Continuous_Q_Actor(
            self.state_dim + self.action_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_action,
            self.use_image
        ).to(self.device)

        self.Critic = Continuous_Q_Critic(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.use_image
        ).to(self.device)

        self.Actor_optimizer = torch.optim.Adam(
            self.Actor.parameters(),
            lr=self.lr,
        )
        self.Critic_optimizer = torch.optim.Adam(
            self.Critic.parameters(),
            lr=self.lr,
        )
        self.temp_optimizer = torch.optim.Adam(
            [self.log_temperature],
              lr=self.lr
        ) 
        self.target_Actor = deepcopy(self.Actor).to(self.device)
        self.target_Critic = deepcopy(self.Critic).to(self.device)
        
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = TAACBuffer(
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
        prev_action = None,
        deterministic = False
    ) -> torch.Tensor:
        with torch.no_grad():
            none_flag = False

            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            if prev_action is None:
                prev_action = np.zeros((state_t.shape[0], self.action_dim))
                none_flag = True

            prev_action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
            if prev_action_t.ndim == 1:
                prev_action_t = prev_action_t.unsqueeze(0)
            actor_input = torch.cat([state_t, prev_action_t], dim=-1)
            new_action = self.Actor(actor_input)


            if not deterministic:
                new_action += torch.normal(0, self.expl_noise, size=new_action.shape).to(self.device)
            new_action = torch.clamp(new_action, -self.max_action, self.max_action)

            if none_flag:
                return new_action.flatten().cpu().numpy(), 1
             
            q_stay = self.Critic(state_t, prev_action_t)
            q_switch = self.Critic(state_t, new_action)
            
            if deterministic:
                beta = (q_stay < q_switch).float().item() # 0: stay, 1: switch
            else:
                q_cat = torch.cat([q_stay, q_switch], dim=-1)
                beta_probs = F.softmax(q_cat/self.temperature, dim=-1)

                dist = torch.distributions.Categorical(beta_probs)
                beta = dist.sample().item()  # 0: stay, 1: switch
            
            action = new_action if beta > 0.5 else prev_action_t
            return action.flatten().cpu().numpy(), beta
        
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
        prev_action,
        reward,
        next_state,
        beta,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            prev_action,
            reward,
            next_state,
            beta,
            done
        )
    
    def update(self, training_steps):

        (
            states, 
            actions, 
            prev_actions,
            rewards, 
            next_states, 
            beta,
            dones,
        ) = self.replay_buffer.sample(self.batch_size, self.seq_len )

        with torch.no_grad():
            next_prev_actions = actions
            target_actor_input = torch.cat([next_states, next_prev_actions], dim=-1)
            next_actions = self.target_Actor(target_actor_input)

            target_Q_stay = self.target_Critic(next_states, next_prev_actions)
            target_Q_switch = self.target_Critic(next_states, next_actions)

            target_Q_values = torch.max(target_Q_stay, target_Q_switch)
            target_beta = (target_Q_switch > target_Q_stay).float()

            bootstrap_mask = ((beta > 0.5) | (target_beta > 0.5) | (dones > 0.5)).float()

            
            target_Q = torch.zeros_like(rewards)
            next_return = target_Q_values[:, -1] 

            for t in reversed(range(self.seq_len)):
                future_val = bootstrap_mask[:, t] * target_Q_values[:, t]*(1 - dones[:, t]) + \
                              (1 - bootstrap_mask[:, t]) * next_return
                current_val = rewards[:, t] + (self.gamma * future_val)
                
                target_Q[:, t] = current_val
                next_return = current_val    
        
        # Critic update
        current_Q = self.Critic(states, actions)
        critic_loss = self.loss_fn(current_Q, target_Q.detach())

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()

        # Actor update
        with torch.no_grad():
            current_q_stay = self.Critic(states, prev_actions)

        new_actions = self.Actor(torch.cat([states, prev_actions], dim=-1))
        current_q_switch = self.Critic(states, new_actions)

        q_cat = torch.cat([current_q_stay, current_q_switch], dim=-1)
        beta_probs = F.softmax(q_cat/self.temperature, dim=-1) # [batch_size, seq_len, 2], beta_probs_shape: torch.Size([64, 10, 2])
        beta_switch = beta_probs[:, :, 1].unsqueeze(-1) 

        # print(f"beta_probs_shape: {beta_probs.shape}, beta_switch_shape: {beta_switch.shape}, current_q_switch_shape: {current_q_switch.shape}")

        actor_loss = -(beta_switch.detach() * current_q_switch).mean()
        
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()

        # temperature update

        alpha_for_grad = self.log_temperature.exp() 
        q_cat_detached = q_cat.detach()
        
        beta_probs_grad = F.softmax(q_cat_detached / alpha_for_grad, dim=-1)
        
        current_entropy = -torch.sum(beta_probs_grad * torch.log(beta_probs_grad + 1e-10), dim=-1).mean()
        
        temperature_loss = (self.log_temperature * (current_entropy - self.target_entropy).detach()).mean()

        self.temp_optimizer.zero_grad()
        temperature_loss.backward()
        self.temp_optimizer.step()

        self.temperature = self.log_temperature.exp().item()
        # print(f"Temperature: {self.log_temperature.exp().item():.4f}")
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