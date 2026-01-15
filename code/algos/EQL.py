from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import action_transform

from .buffer.repetition_buffer import SkipBuffer
from ._modules import Rep_DQN

from torch.func import stack_module_state, functional_call
import copy

class EQL:
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
        alpha,
        fixed_coeff,
        sigma,
        sigma_decay,
        n_sample,
        use_act_skip_buf,
        prev_buffer_save
    ):
        self.base_agent = base_agent
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim
        
        self.is_continuous = False if hasattr(self.base_agent, "n_actions") else True
        
        if self.is_continuous:
            self.n_actions = None
            self.max_action = base_agent.max_action
        else:
            self.n_actions = base_agent.n_actions
            
        
        self.rep_buffer_size: int = rep_buffer_size
        self.rep_batch_size: int = rep_batch_size
        self.hidden_dim: int = base_agent.hidden_dim
        self.update_interval: int = base_agent.update_interval
        self.e_decay: int = e_decay
        self.max_repetition: int = max_repetition
        self.n_sample: int = n_sample

        self.sigma = sigma  
        self.init_sigma = sigma
        self.lr: float = base_agent.lr  
        self.initial_lr: float = base_agent.initial_lr
        self.final_lr: float = base_agent.final_lr
        self.tau: float = base_agent.tau
        self.gamma: float = base_agent.gamma
        self.epsilon: float = max_epsilon
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        
        self.use_image: bool = base_agent.use_image
        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.use_hard_update: bool = base_agent.use_hard_update
        self.fixed_coeff: bool = fixed_coeff
        self.use_act_skip_buf: bool = use_act_skip_buf
        self.sigma_decay: bool = sigma_decay
        

        self.alpha = alpha if self.fixed_coeff else max_epsilon
        self.min_alpha: float = alpha
        
        self.e_greedy_type: str = e_greedy_type
        self.device: str = base_agent.device

        self.max_Rep_Actor = Rep_DQN(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_repetition,
            self.use_image,
            self.n_actions
        ).to(self.device)
        
        self.mean_Rep_Actor = deepcopy(self.max_Rep_Actor)
        
        self.target_max_Rep_Actor = deepcopy(self.max_Rep_Actor)
        self.target_mean_Rep_Actor = deepcopy(self.mean_Rep_Actor)
        
        self.loss_fn = nn.SmoothL1Loss()
        self.max_Rep_Actor_optimizer = optim.Adam(self.max_Rep_Actor.parameters(), lr=self.lr)
        self.mean_Rep_Actor_optimizer = optim.Adam(self.mean_Rep_Actor.parameters(), lr=self.lr)
        
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
        self.alpha_update()

    def lr_decay(self, training_rate):
        self.base_agent.lr_decay(training_rate)
        self.lr = self.base_agent.lr
        
        for param_group in self.max_Rep_Actor_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.mean_Rep_Actor_optimizer.param_groups:
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
                
                max_rep_q = self.max_Rep_Actor(state, action)
                mean_rep_q = self.mean_Rep_Actor(state, action)
                
                if deterministic:
                    # rep_Qs = (1.0 - self.min_alpha) * max_rep_q + self.min_alpha * mean_rep_q
                    rep_Qs = (1.0 - self.alpha) * max_rep_q + self.alpha * mean_rep_q
                    repetitions = torch.argmax(rep_Qs, dim=-1).squeeze().item() + 1
                    return repetitions, rep_Qs
                else:
                    rep_Qs = (1.0 - self.alpha) * max_rep_q + self.alpha * mean_rep_q
                    repetitions = torch.argmax(rep_Qs, dim=-1).squeeze().item() + 1
                    return repetitions

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
            
    def alpha_update(self):
        if not self.fixed_coeff:
            self.alpha = max(self.epsilon, self.min_alpha)
        else:
            self.alpha = self.min_alpha
    
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
            if self.n_actions is not None:
                next_actions = torch.argmax(self.base_agent.target_Actor(next_states), dim=-1, keepdim=True)
                next_actions = action_transform(next_actions, self.n_actions, self.device)
                max_q = self.target_max_Rep_Actor(next_states, next_actions).max(dim=-1, keepdim=True)[0]
                
                mean_q = torch.zeros(self.rep_batch_size, self.n_actions).to(self.device)
                for idx in range(self.n_actions):
                    #  batch x one_hot_action
                    dummy_action = torch.zeros(self.rep_batch_size, self.n_actions).to(self.device)
                    dummy_action[:, idx] = 1.0
                    next_rep_mean_q = self.target_mean_Rep_Actor(next_states, dummy_action).mean(dim=-1, keepdim=False)
                    mean_q[:, idx] = next_rep_mean_q
                
                mean_q  = mean_q.mean(dim=-1, keepdim=True)
            else:
                next_actions = self.base_agent.target_Actor(next_states)
                # max_q = self.target_max_Rep_Actor(next_states, next_actions).max(dim=-1, keepdim=True)[0]
                max_q = self.base_agent.target_Critic(
                    torch.cat([next_states, next_actions], dim=-1)
                )
                
                noises = (torch.randn(self.n_sample, self.rep_batch_size, self.action_dim) * self.sigma).to(self.device)
                # mean_q = torch.zeros(self.rep_batch_size, self.n_sample).to(self.device)
                mean_q = torch.zeros(self.rep_batch_size, self.n_sample).to(self.device)
                for idx, noise in enumerate(noises):
                    noisy_next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
                    # rep_q_values = self.target_mean_Rep_Actor(next_states, noisy_next_actions).mean(dim=-1, keepdim=False)
                    rep_q_values = self.base_agent.target_Critic(
                        torch.cat([next_states, noisy_next_actions], dim=-1)
                    ).squeeze(-1)
                    mean_q[:, idx] = rep_q_values
                    
                mean_q = mean_q.mean(dim=-1, keepdim=True)

            target_max_Q = rewards + not_dones * (self.gamma ** reps) * max_q
            target_mean_Q = rewards + not_dones * (self.gamma ** reps) * mean_q
        
        actions =action_transform(actions, self.n_actions, self.device)
        
        # Update Max Rep Actor
        max_q_values = self.max_Rep_Actor(states, actions).gather(-1, index=rep_idx)
        max_q_loss = self.loss_fn(max_q_values, target_max_Q)
        self.max_Rep_Actor_optimizer.zero_grad()
        max_q_loss.backward()
        self.max_Rep_Actor_optimizer.step()
        
        # Update Mean Rep Actor
        mean_q_values = self.mean_Rep_Actor(states, actions).gather(-1, index=rep_idx)
        mean_q_loss = self.loss_fn(mean_q_values, target_mean_Q)
        self.mean_Rep_Actor_optimizer.zero_grad()
        mean_q_loss.backward()
        self.mean_Rep_Actor_optimizer.step()

        if self.use_hard_update:
            if training_steps % self.update_interval == 0:
                self.target_max_Rep_Actor.load_state_dict(self.max_Rep_Actor.state_dict())
                self.target_mean_Rep_Actor.load_state_dict(self.mean_Rep_Actor.state_dict())
        else:
            for target_param, param in zip(self.target_max_Rep_Actor.parameters(), self.max_Rep_Actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_mean_Rep_Actor.parameters(), self.mean_Rep_Actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        log_dict.update({
            "rep_mean_q_loss": mean_q_loss.clone().cpu().item(),
            "rep_max_q_loss": max_q_loss.clone().cpu().item(),
            "rep_max_td_error": (max_q_values - target_max_Q).mean().clone().cpu().item(),
            "rep_mean_td_error": (mean_q_values - target_mean_Q).mean().clone().cpu().item(),
        })
        return log_dict