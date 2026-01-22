import ale_py
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(ale_py)
gym.register_envs(gymnasium_robotics)

from pprint import pprint
from typing import Any
from omegaconf.dictconfig import DictConfig

from gymnasium.wrappers import (
    AtariPreprocessing, 
    ClipReward, 
    FrameStackObservation
)

from envs import ENVS_REGISTRY

ATARI = "atari"
GRID = "grid"
CLASSIC_CONTROL = "classic"
ROBOTICS = "robotics"
MUJOCO = "mujoco" 
BOX2D = "box2d"

def build_env(
    env_args: DictConfig,
    use_eval_render: bool = False,
) -> gym.Env:
    """
    Build the wrapped environment
    """
    env_type: str = env_args.env_type
    env_name: str = env_args.env_name

    if env_type == GRID:
        env = ENVS_REGISTRY[env_name]()
    elif env_type == BOX2D:
        env = gym.make(
            env_name,
            continuous=True,
            render_mode="rgb_array" if use_eval_render else None,
        )
        
    elif env_type == ATARI:
        env = gym.make(
            env_name,
            render_mode="rgb_array" if use_eval_render else None,
        )
        env = AtariPreprocessing( # 
            env, 
            noop_max= env_args.noop_max,
            terminal_on_life_loss=True,
        )
        action_meanings = env.unwrapped.get_action_meanings()
        print("Action meanings:", action_meanings)
        if action_meanings[1] == "FIRE":
            env = FireResetEnv(env)
        env = FrameStackObservation(
            env, 
            stack_size=env_args.frame_stack
        )
    elif env_type == CLASSIC_CONTROL:
        env = gym.make(
            env_name,
            render_mode="rgb_array" if use_eval_render else None,
        )
    elif env_type == ROBOTICS:
        env = gym.make(
            env_name,
            max_episode_steps=env_args.max_episode_steps,
            render_mode="rgb_array" if use_eval_render else None,
        )
    elif env_type == MUJOCO:
        env = gym.make(
            env_name,
            render_mode="rgb_array" if use_eval_render else None,
        )
    else:
        raise NotImplementedError("Environment type not implemented.")
    return env

def get_env_info(
    env_args: DictConfig,
    env: gym.Env,
    use_step_rate: bool = False
) -> dict[str, Any]:
    """
    Gather environment information: 
        state_dim
        action_dim
        n_actions
    """
    env_type: str = env_args.env_type

    if env_type == GRID:
        state_dim: int = env.observation_space.n
        n_actions: int = env.action_space.n

        env_info: dict[str, Any] = {
            "state_dim": int(state_dim),
            "action_dim": 1,
            "n_actions": int(n_actions),
        }

    elif env_type == ATARI:
        state_dim: tuple[int, int, int] = env.observation_space.shape # C, H, W
        action_dim: int = 1
        n_actions: int = env.action_space.n
        env_info: dict[str, Any] = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "n_actions": n_actions,
            "int_action": True,
        }
    elif env_type == CLASSIC_CONTROL:
        env_info: dict[str, Any] = {}
        is_continuous = isinstance(env.action_space, gym.spaces.Box)
        state_dim: int = env.observation_space.shape[0]
        if use_step_rate:
            state_dim += 1  # add step rate dimension
        
        env_info["state_dim"] = state_dim
        
        if is_continuous:
            action_dim: int = env.action_space.shape[0] 
            n_actions: int = None
            env_info["max_action"] = env.action_space.high[0]
        else:
            action_dim: int = 1
            n_actions: int = env.action_space.n
        
        env_info["action_dim"] = action_dim
        env_info["n_actions"] = n_actions
    
    elif env_type == ROBOTICS:
        env_info: dict[str, Any] = {}
        obs_space = env.observation_space
        state_dim: int = obs_space["observation"].shape[0] + obs_space["desired_goal"].shape[0] + obs_space["achieved_goal"].shape[0]
        if use_step_rate:
            state_dim += 1  # add step rate dimension
        env_info["state_dim"] = state_dim
        action_dim: int = env.action_space.shape[0]

        env_info["action_dim"] = env.action_space.shape[0]
        env_info["max_action"] = float(env.action_space.high[0])
        env_info["n_actions"] = None
        env_info["use_log_reward"] = env_args.use_log_reward

    elif env_type == MUJOCO:
        env_info: dict[str, Any] = {}
        state_dim: int = env.observation_space.shape[0]
        if use_step_rate:
            state_dim += 1  # add step rate dimension
        env_info["state_dim"] = state_dim
        action_dim: int = env.action_space.shape[0]

        env_info["action_dim"] = env.action_space.shape[0]
        env_info["max_action"] = float(env.action_space.high[0])
        env_info["n_actions"] = None
    else:
        raise NotImplementedError(f"Environment type: {env_type} not implemented.")
            
    print("*"*20, "Environment Info", "*"*20)
    pprint(env_info, width=1)
    print("*"*60)
    return env_info

class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated,info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        # obs, _, terminated, truncated, info = self.env.step(2)
        # if terminated or truncated:
        #     self.env.reset(**kwargs)
        return obs, info
