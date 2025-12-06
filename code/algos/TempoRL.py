import torch
import torch.nn as nn
import torch.optim as optim

from ._modules import Rep_DuelDQN, Rep_DQN
from .buffer.repetition_buffer import SkipBuffer

from utils.utils import action_transform

class TempoRL:
    def __init__(
        self,
        base_agent,
        max_repetition,
        e_greedy_type,
        e_decay,
        max_epsilon,
        min_epsilon,
        use_dueling,
    ):
        self.base_agent = base_agent
        self.state_dim: int = base_agent.state_dim
        self.action_dim: int = base_agent.action_dim
        
        if hasattr(self.base_agent, "n_actions"):
            self.n_actions = base_agent.n_actions
        else:
            self.n_actions = None
            
        self.buffer_size: int = base_agent.buffer_size
        self.batch_size: int = base_agent.batch_size
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
        self.max_grad_norm: float = base_agent.max_grad_norm
        
        self.e_greedy_type: str = e_greedy_type
        self.device: str = base_agent.device
        
        self.use_image: bool = base_agent.use_image
        self.use_ddqn: bool = base_agent.use_ddqn
        self.use_dueling: bool = use_dueling
        self.use_lr_decay: bool = base_agent.use_lr_decay
        self.use_hard_update: bool = base_agent.use_hard_update
        
        if self.use_dueling:
            self.Rep_Actor = Rep_DuelDQN(
                self.state_dim,
                self.action_dim,
                self.hidden_dim,
                self.max_repetition,
                self.use_image,
                self.n_actions
            ).to(self.device)
        else:
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
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
    def epsilon_decay(self, training_steps):
        if hasattr(self.base_agent, "epsilon_decay"):
            self.base_agent.epsilon_decay(training_steps)

        training_steps = torch.tensor(training_steps, dtype=torch.float32)
        if self.e_greedy_type == "exponential":
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
                rep_q_values = self.Rep_Actor(state, action).cpu()
                repetitions = torch.argmax(rep_q_values, dim=-1).squeeze().item() + 1
        else:
            repetitions = torch.randint(1, self.max_repetition + 1, (1,)).item()

        return repetitions

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        skip_states,
        skip_rewards
    ):
        self.base_agent.add(
            state,
            action,
            reward,
            next_state,
            done
        )
        
        for idx, start_state in enumerate(skip_states):
            skip_reward = 0
            skip_step = 0
            for exp, r in enumerate(skip_rewards[idx:]):
                skip_reward += (self.gamma ** exp) * r
                skip_step += 1

            self.rep_replay_buffer.add(
                state = start_state,
                action = action,
                repetition = skip_step,
                reward = skip_reward,
                next_state = next_state,
                done = done
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
        ) = self.rep_replay_buffer.sample(self.batch_size)
        
        rep_idx = reps.long() - 1
        
        with torch.no_grad():
            if self.use_ddqn:
                next_actions = self.base_agent.Actor(next_states).argmax(dim=-1, keepdim=True)
                next_q_values = self.base_agent.target_Actor(next_states).gather(dim=-1, index=next_actions)
            else:
                next_actions = self.base_agent.target_Actor(next_states)
                next_q_values = torch.max(next_actions, dim=-1, keepdim=True)[0]
                
            target_q_values = rewards + not_dones * (self.gamma ** reps) * next_q_values

        actions =action_transform(actions, self.n_actions, self.device)
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
                
        
        