import torch
import torch.nn as nn

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
    
class DuelDQN(nn.Module):
    def __init__(
        self, 
        state_dim, 
        n_actions, 
        hidden_dim, 
        use_image
    ):
        super(DuelDQN, self).__init__()
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
            
            self.v = nn.Sequential(
                nn.Linear(cnn_output_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.adv = nn.Sequential(
                nn.Linear(cnn_output_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
        else:
            state_dim = state_dim[0]
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.adv = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )   

    def forward(self, x):
        if self.use_image:
            x = self.cnn(x)
            v = self.v(x)
            adv = self.adv(x)
        else:
            x = self.encoder(x)
            v = self.v(x)
            adv = self.adv(x)
        x = v + (adv - adv.mean(dim=-1, keepdim=True))
        return x
    
class Rep_DuelDQN(nn.Module):
    def __init__(
        self, 
        state_dim,
        action_dim,
        hidden_dim,
        max_repetition,
        use_image,
        n_actions
    ):
        super(Rep_DuelDQN, self).__init__()
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
            )
            self.v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.adv = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.adv = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )   
        
    def forward(self, state, actions):
        if self.use_image:
            x = self.cnn(state)
            x = torch.cat([x, actions], dim=-1)
            x = self.mixing_layer(x)
            v = self.v(x)
            adv = self.adv(x)
        else:
            x = torch.cat([state, actions], dim=-1)
            x = self.encoder(x)
            v = self.v(x)
            adv = self.adv(x)
        x = v + (adv - adv.mean(dim=-1, keepdim=True))
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
    
    
class Duel_Ensemble_Net(nn.Module):
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
        super(Duel_Ensemble_Net, self).__init__()
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
            )
            
            v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            adv = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
        else:
            encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            )
            
            mixing_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            
            v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            adv = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max_repetition)
            )
            
        return nn.ModuleDict({
            "encoder": encoder,
            "mixing_layer": mixing_layer,
            "v": v,
            "adv": adv
        })
    
    def forward(self, state, action):
        outputs = []
        for model in self.models:
            if self.use_image:
                x = model["encoder"](state)
                x = torch.cat([x, action], dim=-1)
                x = model["mixing_layer"](x)
                v = model["v"](x)
                adv = model["adv"](x)
            else:
                x = torch.cat([state, action], dim=-1)
                x = model["encoder"](x)
                x = model["mixing_layer"](x)
                v = model["v"](x)
                adv = model["adv"](x)
            
            q_values = v + (adv - adv.mean(dim=-1, keepdim=True))
            outputs.append(q_values)
        
        return torch.stack(outputs, dim=0)
    

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
            