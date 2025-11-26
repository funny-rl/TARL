import torch
import random
import ale_py
import numpy as np
import gymnasium as gym 
gym.register_envs(ale_py)

from pprint import pprint
from typing import Tuple, Dict, Any

from contextlib import contextmanager

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


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

device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATARI_ENVS: list[str] = [
    "PongNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "ALE/Pong-v5",
]

CLASSIC_ENVS: list[str] = [
    "CartPole-v1",
    "MountainCar-v0",
]

ENV_INFO: Dict[str, Any] = {}

def build_env(env_name, use_step_rate, args) -> gym.Env:
    """Build the environment and gather environment information."""
    if env_name in ATARI_ENVS:
        env_args = args.envs
        
        env = gym.make(env_name, repeat_action_probability=0.0, frameskip=1)
        env = AtariPreprocessing(
            env, 
            grayscale_obs=env_args.grayscale_obs, 
            grayscale_newaxis=env_args.grayscale_newaxis,
            terminal_on_life_loss=env_args.terminal_on_life_loss,
            screen_size=env_args.screen_size,
            scale_obs=env_args.scale_obs, 
            frame_skip=env_args.frame_skip
        )
        env = FrameStackObservation(env, env_args.num_stack)
    else:
        env = gym.make(env_name)
    
    if not hasattr(build_env, "initialized"):
        global ENV_INFO
        if env_name in ATARI_ENVS:
            state_dim: Tuple[int] = env.observation_space.shape # H, W, C
            
            ENV_INFO["state_dim"] = state_dim
            ENV_INFO["action_dim"] = 1
            ENV_INFO["n_actions"] = int(env.action_space.n)
        elif env_name in CLASSIC_ENVS:
            state_dim: int = env.observation_space.shape[-1]
            ENV_INFO["state_dim"] = state_dim + 1 if use_step_rate else state_dim
            ENV_INFO["action_dim"] = 1
            ENV_INFO["n_actions"] = int(env.action_space.n)
        else:
            raise NotImplementedError(f"Environment {env_name} is not supported.")
        print("*"*20, "Environment Info", "*"*20)
        pprint(ENV_INFO, width=1)
        print("*"*60)
        build_env.initialized = True
        
    return env

def get_model_configs(args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get algorithm and model configurations from args."""
    algo_name: str = args.algos.name
    algo_args = args.algo_args
    
    lr: float = algo_args.lr
    gamma: float = algo_args.gamma
    buffer_size: int = algo_args.buffer_size
    batch_size: int = algo_args.batch_size
    e_greedy_type: str = algo_args.e_greedy_type
    min_epsilon: float = algo_args.min_epsilon
    data_type: str = algo_args.data_type
    hidden_dim: int = algo_args.hidden_dim
    target_update_rate: int = algo_args.target_update_rate

    if algo_name == "DQN":
        algo_dict: Dict[str, Any] = {
            "state_dim": ENV_INFO["state_dim"],
            "action_dim": ENV_INFO["action_dim"],
            "n_actions": ENV_INFO["n_actions"],
            "lr": lr,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "e_greedy_type": e_greedy_type,
            "min_epsilon": min_epsilon,
            "data_type": data_type,   
            "target_update_rate": target_update_rate,
            "use_ddqn": args.algos.use_ddqn,
            "device": device
        }
    else:
        raise NotImplementedError(f"Algorithm {algo_name} is not supported.")

    model_args = args.algos.get("models", None)

    if model_args is None:
        return algo_dict, {}
    else:
        model_name = model_args.name
        if model_name == "TempoRL":
            model_dict: Dict[str, Any] = {
                "max_repetition" : model_args.max_repetition,
                "e_greedy_type" : e_greedy_type,
                "min_epsilon" : min_epsilon,
            }
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")

        return algo_dict, model_dict

@contextmanager
def episode_stats(stats: dict):
    backup = stats.copy()
    try:
        yield stats
    finally:
        stats.clear()
        stats.update(backup)

def state_transform(state, use_step_rate: bool, env: gym.Env):
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

def to_wandb_name(s: str) -> str:
    bad_chars = "/\\#?%:"
    for c in bad_chars:
        s = s.replace(c, "_")
    return s