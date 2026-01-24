import torch
import torch.nn as nn
import torch.optim as optim

from ._modules import Rep_DQN
from .buffer.repetition_buffer import SkipBuffer

from utils.utils import action_transform

class TempoRL:
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
        
        self.lr: float = base_agent.lr
        self.initial_lr: float = base_agent.initial_lr
        self.final_lr: float = base_agent.final_lr
        
        self.tau: float = base_agent.tau
        self.gamma: float = base_agent.gamma
        self.epsilon: float = max_epsilon
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        
        self.e_greedy_type: str = e_greedy_type
        self.device: str = base_agent.device
        
        self.use_image: bool = base_agent.use_image
        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.use_hard_update: bool = base_agent.use_hard_update
        self.use_act_skip_buf: bool = use_act_skip_buf

        self.Rep_Actor = Rep_DQN(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.max_repetition,
            self.use_image,
            self.n_actions
        ).to(self.device)
        
        self.loss_func = nn.SmoothL1Loss()
        self.Rep_Actor_optimizer = optim.Adam(
            self.Rep_Actor.parameters(),
            lr=self.lr
        )
        
        self.rep_replay_buffer = SkipBuffer(
            buffer_size=self.rep_buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            prev_buffer_save=prev_buffer_save,
            device=self.device
        )
        
        self.base_num_parameters = sum(p.numel() for p in self.base_agent.Actor.parameters() if p.requires_grad)
        print(f"[{self.base_agent.__class__.__name__}] Number of parameters: {self.base_num_parameters}")
        self.rep_num_parameters = sum(p.numel() for p in self.Rep_Actor.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] Number of parameters: {self.rep_num_parameters}")   

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
                rep_Qs = self.Rep_Actor(state, action).cpu()
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
            not_dones,
        ) = self.rep_replay_buffer.sample(self.rep_batch_size)
        
        rep_idx = reps.long() - 1
        
        with torch.no_grad():
            if self.is_continuous:
                next_actions = self.base_agent.target_Actor(next_states)
                next_q_values = self.base_agent.target_Critic(next_states, next_actions)
            else:   
                next_actions = self.base_agent.target_Actor(next_states)
                next_q_values = torch.max(next_actions, dim=-1, keepdim=True)[0]
            
            target_q_values = rewards + not_dones * (self.gamma ** reps) * next_q_values

        actions = action_transform(actions, self.n_actions, self.device)
        q_values = self.Rep_Actor(states, actions).gather(-1, index=rep_idx)
        q_loss = self.loss_func(q_values, target_q_values)

        self.Rep_Actor_optimizer.zero_grad()
        q_loss.backward()
        self.Rep_Actor_optimizer.step()

        log_dict.update({
            "rep_q_loss": q_loss.clone().cpu().item(),
            "rep_td_error": (q_values - target_q_values).mean().clone().cpu().item(),
        })
        return log_dict
                
    def save_model(self, path: str):
        torch.save(self.Rep_Actor.state_dict(), path + "rep_actor.pth")
        self.base_agent.save_model(path)
        