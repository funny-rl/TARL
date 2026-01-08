import torch

class Random:
    def __init__(
        self, 
        n_actions,
        action_dim,
        max_action
    ):
        self.n_actions: int = n_actions
        self.action_dim: int = action_dim
        self.max_action: float = max_action
        self.lr = 0.0
        
    def epsilon_decay(self, training_steps):
        pass

    def lr_decay(self, training_rate):
        pass
    
    def select_action(
        self,
        state: torch.Tensor, 
        deterministic: bool = False,
    ) -> int:
        if self.n_actions is not None:
            action = torch.randint(0, self.n_actions, (1,))
        else:
            action = torch.empty(self.action_dim).uniform_(-self.max_action, self.max_action)
            
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
        pass
        
    def update(self, training_steps: int) -> dict:
        return {}

    def save_model(self, path: str):
        raise NotImplementedError("U can't save the random agent.")