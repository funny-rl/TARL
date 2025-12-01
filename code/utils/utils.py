import torch
import random
import numpy as np
import gymnasium as gym
import torch.nn.functional as F

from typing import Any

from contextlib import contextmanager

def set_seed(seed: int) -> None:
    """Seed the program."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
@contextmanager
def episode_stats(stats: dict):
    """Context manager to handle episode statistics."""
    backup = stats.copy()
    try:
        yield stats
    finally:
        stats.clear()
        stats.update(backup)
        
def state_transform(
    state: Any, 
    use_step_rate: bool, 
    env: gym.Env,
    device: str
) -> torch.Tensor:
    """Transform the state based on whether step rate is used."""
    if isinstance(state, torch.Tensor):
        state_tensor = state.clone().detach()
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32)
    
    if state_tensor.dim() == 1:
        if use_step_rate:
            step_rate = torch.tensor([env._elapsed_steps / env._max_episode_steps])
            state_tensor = torch.cat([state_tensor, step_rate], dim=-1).unsqueeze(0)
    
    elif state_tensor.dim() == 3:
        if state_tensor.max() > 1.0 + 1e-6:
            state_tensor = state_tensor / 255.0
        state_tensor = state_tensor.unsqueeze(0)  #B, C, H, W

    return state_tensor.to(device)


def action_transform(
    action: Any, 
    n_actions: int | None,
    device: str
) -> torch.Tensor:
    """Transform the action to a tensor."""
    if isinstance(action, torch.Tensor):
        action_tensor = action.clone().detach() # (bs x 1)
        if n_actions is not None:
            action_tensor = action_tensor.long()
            if action_tensor.dim() == 2:
                action_tensor = F.one_hot(action_tensor, num_classes=n_actions).squeeze(1).float()
                
    elif type(action) == int and n_actions is not None:
        action_tensor = torch.tensor(action, dtype=torch.int64)
        if action_tensor.dim() == 0:
            one_hot = F.one_hot(action_tensor, num_classes=n_actions)
            action_tensor = one_hot.unsqueeze(0)
        else:
            raise ValueError("Not supported action shape for discrete action space.")
    else:
        raise ValueError("Unsupported action type.")

    return action_tensor.to(device)