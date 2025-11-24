import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .replay_buffer.repetition_buffer import RepetitionBuffer



class EnsembleNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_repetitions, hidden_dim, n_heads):
        super(EnsembleNet, self).__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_repetitions)
        ) for _ in range(self.n_heads)])

    def forward_single_head(self, x, k):
        x = self.heads[k](x)
        return x

    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out

class UTE:
    def __init__(
        self, 
        base_agent,
        max_repetitions,
        uncertainty_factor,
        n_heads,
        e_greedy_type,
        power_value,
        min_epsilon
    ):
        self.ba = base_agent
        self.n_heads: int = n_heads
        self.state_dim: int = self.ba.state_dim
        self.action_dim: int = self.ba.action_dim
        self.batch_size: int = self.ba.batch_size
        self.hidden_dim: int = self.ba.hidden_dim
        self.buffer_size: int = self.ba.buffer_size 
        self.max_repetitions: int = max_repetitions
        self.bernoulli_probability = 0.5  # Probability for Bernoulli mask in ensemble training

        self.lr: float = self.ba.lr
        self.epsilon: float = 1.0
        self.power_value: float = power_value
        self.min_epsilon: float = min_epsilon
        self.uncertainty_factor: float = uncertainty_factor
        self.discount_factor: float = self.ba.discount_factor
        
        self.device: str = self.ba.device
        self.e_greedy_type: str = e_greedy_type
        
        self.has_ba_epsilon: bool = hasattr(self.ba, 'epsilon')

        if hasattr(self.ba, 'n_actions'):
            self.action_scaling = self.ba.n_actions
        elif hasattr(self.ba, 'action_dim'):
            self.action_scaling = self.ba.max_action
        else:
            raise ValueError("Base agent must have either 'n_actions' or 'action_dim' attribute.")
        
        self.ensemble_net = EnsembleNet(
            self.state_dim, 
            self.action_dim,
            self.max_repetitions, 
            self.hidden_dim,
            self.n_heads
        ).to(self.device)
        
        self.ensemble_net_optimizer = torch.optim.Adam(self.ensemble_net.parameters(), lr=self.lr)
        
        self.repetition_buffer = RepetitionBuffer(
            max_size = self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        


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
        for param_group in self.ensemble_net_optimizer.param_groups:
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
                outputs = []
                repetition_logits_heads = self.ensemble_net(input)
                for head_output in repetition_logits_heads:
                    outputs.append(head_output.cpu().numpy())
                mean_Q = np.mean(outputs, axis=0)
                std_Q = np.std(outputs, axis=0)
                Q_tilda = mean_Q + self.uncertainty_factor*std_Q
                repetition = Q_tilda.argmax().item() + 1
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
        inputs = torch.cat(
            [
                states,
                actions / self.action_scaling
            ], dim=-1
        ).to(self.device)
        current_outputs = self.ensemble_net(inputs)  # Forward pass to ensure gradients are tracked
        masks = torch.bernoulli(torch.zeros((self.batch_size, self.n_heads), device=self.device) + self.bernoulli_probability)

        cnt_losses = 0.0
        rep_td_error = 0.0
        
        for k in range(self.n_heads):
            num_update = masks[:, k].sum()
            if num_update > 0:
                current_Q = current_outputs[k].gather(1, (reps - 1).long())
                loss = F.mse_loss(current_Q * masks[:, k], target_Q * masks[:, k]) / num_update
                cnt_losses += loss
                rep_td_error += (target_Q - current_Q).mean().item() / num_update
        
        skip_losses = cnt_losses / self.n_heads
        rep_td_error = rep_td_error / self.n_heads
        
        self.ensemble_net_optimizer.zero_grad()
        skip_losses.backward()
        self.ensemble_net_optimizer.step()
        
        log_dict = {
            "rep_loss": skip_losses.cpu().item(),
            "rep_td_error": rep_td_error.cpu().item(),
        } | parent_log
        
        return log_dict
        
        
    