import torch
import torch.nn as nn

class Continuous_Q_Actor(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim,
        max_action,
        use_image
    ):
        super(Continuous_Q_Actor, self).__init__()
        self.max_action = max_action
        self.use_image = use_image
        if self.use_image:
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
                nn.Linear(hidden_dim, action_dim), nn.Tanh()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim), nn.Tanh()
            )
    def forward(self, x):
        if self.use_image:
            x = self.cnn(x)
        return self.max_action * self.fc(x)

class Continuous_Q_Critic(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim,
        use_image
    ):
        super(Continuous_Q_Critic, self).__init__()
        self.use_image = use_image
        if self.use_image:
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
                nn.Linear(cnn_output_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, states, actions):
        if self.use_image:
            x = self.cnn(states)
            x = torch.cat([x, actions], dim=-1)
        else:
            x = torch.cat([states, actions], dim=-1)
            
        q_value = self.fc(x)
        return q_value

class QNet(nn.Module):
    def __init__(
        self, 
        state_dim, 
        n_actions, 
        hidden_dim, 
        use_image
    ):
        super(QNet, self).__init__()
        self.use_image = use_image
        if self.use_image:
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
                nn.Linear(hidden_dim, n_actions)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
    def forward(self, x):
        if self.use_image:
            x = self.cnn(x)
        x = self.fc(x)
        return x
    
class Rep_DQN(nn.Module):
    def __init__(
        self, 
        state_dim,
        action_dim,
        hidden_dim,
        max_repetition,
        use_image,
        n_actions = None
    ):
        super(Rep_DQN, self).__init__()
        self.use_image = use_image
        
        if n_actions is not None:
            action_dim = n_actions
        
        if self.use_image:
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
            
            self.mixing_layer = nn.Sequential(
                nn.Linear(cnn_output_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
        
    def forward(self, state, actions):
        if self.use_image:
            x = self.cnn(state)
            x = torch.cat([x, actions], dim=-1)
            x = self.mixing_layer(x)
        else:
            x = torch.cat([state, actions], dim=-1)
            x = self.encoder(x)
        return x

class Ensemble_DQN(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim, 
        max_repetition,
        use_image,
        num_ensemble,
        n_actions
    ):
        super(Ensemble_DQN, self).__init__()
        self.use_image = use_image
        
        if n_actions is not None:
            action_dim = n_actions
        
        self.models = nn.ModuleList(
            [
                self._create_model(
                    state_dim,
                    action_dim,
                    hidden_dim,
                    max_repetition
                    
                ) for _ in range(num_ensemble)
            ]
        )
        
    def _create_model(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        max_repetition
    ):
        if self.use_image:
            C_dim = state_dim[0]
            h_dim = state_dim[1]
            w_dim = state_dim[2]
            
            encoder = nn.Sequential(
                nn.Conv2d(C_dim, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            test_tensor = torch.zeros((1, C_dim, h_dim, w_dim))
            with torch.no_grad():
                cnn_output_dim = encoder(test_tensor).shape[-1]
                
            mixing_layer = nn.Sequential(
                nn.Linear(cnn_output_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
        else:
            encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),

            )
            mixing_layer = nn.Sequential(                
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
        
        return nn.ModuleDict({
            "encoder": encoder,
            "mixing_layer": mixing_layer
        })
    
    def forward(self, state, action):
        outputs = []
        for model in self.models:
            if self.use_image:
                x = model["encoder"](state)
                x = torch.cat([x, action], dim=-1)
                x = model["mixing_layer"](x)
            else:
                x = torch.cat([state, action], dim=-1)
                x = model["encoder"](x)
                x = model["mixing_layer"](x)
            outputs.append(x)
            
        return torch.stack(outputs, dim=0)
