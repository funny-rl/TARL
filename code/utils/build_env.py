import ale_py
import numpy as np
import gymnasium as gym
gym.register_envs(ale_py)
from gymnasium.envs.registration import registry

from pprint import pprint
from typing import Any
from omegaconf.dictconfig import DictConfig

from gymnasium.wrappers import (
    AtariPreprocessing, 
    ClipReward, 
    FrameStackObservation
)

ATARI = "ale_py.env:AtariEnv"

def is_atari_env(env_name: str) -> bool:
    try:
        spec = registry[env_name]
        return "atari" in (spec.entry_point or "").lower()
    except KeyError:
        return False

def build_env(
    env_name: str,
    env_args: DictConfig,
    use_eval_render: bool = False,
) -> gym.Env:
    """
    Build the wrapped environment
    """
    entry_point: str = registry[env_name].entry_point
    if entry_point == ATARI:
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

    else:
        raise NotImplementedError("Environment type not implemented.")
    return env

def get_env_info(
    env_name: str,
    env: gym.Env
) -> dict[str, Any]:
    """
    Gather environment information: 
        state_dim
        action_dim
        n_actions
    """
    entry_point: str = registry[env_name].entry_point
    if entry_point == ATARI:
        state_dim: tuple[int, int, int] = env.observation_space.shape # C, H, W
        action_dim: int = 1
        n_actions: int = env.action_space.n
        env_info: dict[str, Any] = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "n_actions": n_actions
        }
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
