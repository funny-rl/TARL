import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

from .buffer.naive_buffer import NaiveReplayBuffer

class QNet(nn.Module):
    def __init__(
        self, 
        state_dim, 
        n_actions, 
        hidden_dim, 
        data_type
    ):
        super(QNet, self).__init__()
        self.data_type = data_type
        if self.data_type == "image":
            C_dim = state_dim[0]
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
                cnn_output_dim = self.cnn(self.test_tensor).shape[-1]
            
            self.fc = nn.Sequential(
                nn.Linear(cnn_output_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
        elif self.data_type == "feature":
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
        if self.data_type == "image":
            x = self.cnn(x)
        x = self.fc(x)
        return x

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
        min_epsilon,
        data_type,
        target_update_rate,
        use_ddqn,
        device
    ):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.n_actions: int = n_actions
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.hidden_dim: int = hidden_dim
        
        self.lr: float = lr
        self.epsilon: float = 1.0
        self.gamma: float = gamma
        self.min_epsilon: float = min_epsilon
        self.initial_lr: float = self.lr
        self.final_lr: float = 0.1 * self.initial_lr
        self.power_value: float = 10.0
        self.tau: float = target_update_rate
        
        self.use_ddqn: bool = use_ddqn
        
        self.e_greedy_type: str = e_greedy_type
        self.data_type: str = data_type
        self.device: str = device
        
        self.Actor = QNet(
            self.state_dim, 
            self.n_actions, 
            self.hidden_dim,
            self.data_type,
        ).to(self.device)
        self.Actor_optimizer = optim.Adam(
            self.Actor.parameters(), 
            lr=self.lr
        )
        self.target_Actor = deepcopy(self.Actor).to(self.device)
            
        self.replay_buffer = NaiveReplayBuffer(
            buffer_size=self.buffer_size,
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
        
    def lr_decay(self, training_rate):
        training_rate = torch.clamp(
            torch.tensor(training_rate, dtype=torch.float32),
            0.0, 1.0
        )
        cosine = 0.5 * (1 + torch.cos(torch.pi * training_rate))
        self.lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine.item()
        for param_group in self.Actor_optimizer.param_groups:
            param_group['lr'] = self.lr

    def select_action(
        self,
        state, 
        deterministic = False,
    ):
        if deterministic or torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.Actor(state)
                action: int = q_values.argmax(dim=-1).item()  
        else:
            action: int = torch.randint(0, self.n_actions, ()).item()  

        return action
    
    def select_repetition(
        self, 
        state,
        action = None,
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
        skip_states,
        skip_rewards
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
    )
        
    def update(self):
        (
            states, 
            actions, 
            rewards, 
            next_states, 
            not_dones,
        ) = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            if self.use_ddqn:
                next_actions = self.Actor(next_states).argmax(dim=-1, keepdim=True)
                next_q_values = self.target_Actor(next_states).gather(1, next_actions)
            else:
                next_q_values = torch.max(self.target_Actor(next_states), dim = -1, keepdim = True)[0]
            target_q_values = rewards + self.gamma * not_dones * next_q_values
        
        current_q_values = self.Actor(states).gather(1, actions.long())        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.Actor_optimizer.zero_grad()
        loss.backward()
        self.Actor_optimizer.step()
        
        # Soft update of target network
        for target_param, param in zip(self.target_Actor.parameters(), self.Actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        log_dict: dict[str, float] = {
            "td_error": (target_q_values - current_q_values).cpu().mean().item(),
            "q_loss": loss.cpu().item()
        }
        return log_dict