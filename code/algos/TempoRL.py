import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer.repetition_buffer import RepetitionBuffer

class RepetitionNet(nn.Module): 
    def __init__(self, state_dim, action_dim, max_repetitions, hidden_dim):
        super(RepetitionNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, max_repetitions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TempoRL:
    def __init__(
        self, 
        base_agent,
        max_repetitions,
        e_greedy_type,
        power_value,
        min_epsilon
    ):
        self.ba = base_agent
        
        self.batch_size: int = self.ba.batch_size
        self.max_repetitions: int = max_repetitions  
        
        self.lr: float = self.ba.lr
        self.epsilon: float = 1.0
        self.power_value: float = power_value
        self.discount_factor: float = self.ba.discount_factor
        self.min_epsilon: float = min_epsilon
        
        self.device: str = self.ba.device
        self.e_greedy_type: str = e_greedy_type
        
        self.repetition_net = RepetitionNet(
            self.ba.state_dim, 
            self.ba.action_dim, 
            self.max_repetitions, 
            self.ba.hidden_dim
        ).to(self.ba.device)

        self.repetition_net_optimizer = torch.optim.Adam(
            self.repetition_net.parameters(), 
            lr=self.ba.lr
        )

        self.repetition_buffer = RepetitionBuffer(
            max_size = self.ba.buffer_size,
            state_dim=self.ba.state_dim,
            action_dim=self.ba.action_dim,
            device=self.ba.device
        )
        
        self.has_ba_epsilon: bool = hasattr(self.ba, 'epsilon')
        
        if hasattr(self.ba, 'n_actions'):
            self.action_scaling = self.ba.n_actions
        elif hasattr(self.ba, 'action_dim'):
            self.action_scaling = self.ba.max_action
        else:
            raise ValueError("Base agent must have either 'n_actions' or 'action_dim' attribute.")
            
        
    def epsilon_decay(self, training_rate):
        if self.has_ba_epsilon:
            self.ba.epsilon_decay(training_rate)
            self.epsilon = self.ba.epsilon
        else:
            if self.e_greedy_type == "linear":
                self.epsilon = min(1.0, max(self.min_epsilon, 1 - training_rate))
            elif self.e_greedy_type == "power":
                self.epsilon = (min(1.0, max(self.min_epsilon, (1.0 - training_rate) ** self.power_value)))
            else:
                raise ValueError(f"e_greedy_type {self.e_greedy_type} is not supported.")

        
    def lr_decay(
        self, 
        training_rate,
    ):
        self.ba.lr_decay(training_rate)
        self.lr = self.ba.lr
        for param_group in self.repetition_net_optimizer.param_groups:
            param_group['lr'] = self.lr

    def select_action(
        self,
        state,
        deterministic = False  
    ): 
        return self.ba.select_action(
            state,
            deterministic
        )
        
    def select_repetition(
        self,
        state,
        action,
        warmup = False,
        deterministic = False  
    ):
        if (warmup or torch.rand(1).item() < self.epsilon) and not deterministic:
            return np.random.randint(1, self.max_repetitions + 1)
        else:
            with torch.no_grad():
                action = torch.tensor(action)
                if action.dim() == 0:
                    action = action.view(1, -1).float()
                if action.dim() == 1:
                    action = action.unsqueeze(0).float()
                    
                input = torch.concat(
                    [
                        state, 
                        torch.tensor(action / self.action_scaling).to(self.device)
                    ], dim=-1 
                )
                repetition_logits = self.repetition_net(input).squeeze(0).cpu()
                repetition = repetition_logits.argmax().item() + 1
                return repetition
        
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
        self.ba.add(
            state,
            action, 
            reward,
            next_state, 
            done, 
        )
        for idx, start_state in enumerate(skip_states):
            skip_reward = 0
            skip_step = 0
            for exp, r in enumerate(skip_rewards[idx:]):
                skip_reward += (self.discount_factor ** exp) * r
                skip_step += 1
                
            self.repetition_buffer.add(
                state = start_state,
                action = action,
                rep = skip_step,
                reward = skip_reward,
                next_state = next_state,
                done = done
            ) 
        
    def update(self):
        parent_log = self.ba.update()
        (
            states, 
            actions, 
            reps, 
            rewards, 
            next_states, 
            dones
        ) = self.repetition_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            target_Qs = self.ba.target_Actor(next_states)
            target_Q = torch.max(target_Qs, dim = -1, keepdim = True)[0]
            target_Q = rewards + (1 - dones) * (self.discount_factor ** reps) * target_Q
        
        inputs = torch.concat(
            [
                states,
                actions / self.action_scaling
            ], dim=-1
        ).to(self.device)
        
        n_step_RQ = self.repetition_net(inputs).gather(1, (reps - 1).long())
        rep_loss = F.mse_loss(n_step_RQ, target_Q)
        rep_td_error = (target_Q - n_step_RQ).mean().item()
        
        self.repetition_net_optimizer.zero_grad()
        rep_loss.backward()
        self.repetition_net_optimizer.step()
        
        log_dict = {
            "rep_loss": rep_loss.cpu().item(),
            "rep_td_error": rep_td_error,
        } | parent_log
        
        return log_dict
        
    
        
        
        
        