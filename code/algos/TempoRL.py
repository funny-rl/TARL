import torch

class TempoRL:
    def __init__(
        self, 
        action_agent,
        max_repetition,
        e_greedy_type,
        min_epsilon
    ):
        pass
    
    def select_repetition(
        self, 
        state,
        action = None,
        deterministic = False
    ):
        if not deterministic and action is None:
            repetition = torch.randint(1, self.n_actions + 1, ()).item()
        else:
            pass