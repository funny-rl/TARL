import numpy as np
import sys
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from matplotlib.patches import Rectangle
from gymnasium import spaces
from io import StringIO
from typing import Tuple

import time
from scipy.spatial.distance import cityblock

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

class GridCore(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape: Tuple[int] = (5, 10), start: Tuple[int] = (0, 0),
                 goal: Tuple[int] = (0, 9), max_steps: int = 100,):
        
        super().__init__()
        try:
            self.shape = self._shape
        except AttributeError:
            self.shape = shape
        self.state_size = np.prod(self.shape, dtype=int)  # type: int
        self.action_size = 4
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self._steps = 0
        self.total_steps = 0

        self.init_state = np.zeros(self.state_size)
        self.init_state[np.ravel_multi_index(self.start, self.shape)] = 1.0

        self.P = self._init_transition_probability()
        self.observation_space = spaces.Discrete(self.state_size)
        self.action_space = spaces.Discrete(self.action_size)
        self.one_hot_obs = np.eye(self.state_size, dtype=np.float32)

        self.visited = np.zeros(self.shape)


    def step(self, action):
        self._steps += 1

        transitions = self.P[self.s][action]
        i = self.categorical_sample([t[0] for t in transitions], self.np_random)
        prob, state, reward, done = transitions[i]

        y,x = np.unravel_index(state, self.shape)
        self.visited[y][x] += 1

        # 상태 업데이트
        self.s = state
        self.lastaction = action

        terminated = done  
        truncated = False
        
        # 시간 제한 체크 
        if self._steps >= self.max_steps:
            truncated = True

        if terminated or truncated:
            fig, ax = plt.subplots(figsize=(self.shape[1], self.shape[0]))
            log_visited = np.log10(self.visited+1)

            sns.heatmap(
                log_visited, 
                ax=ax, 
                cmap="Blues",       # 파란색 계열 컬러맵
                cbar=True,          # 컬러바 표시
                linewidths=.5,
                linecolor='black',
                square=True
            )
            ax.set_title(f"Coverage Plot")
            for pit in self.pits:
                # print(self.pits) 
                r, c = np.unravel_index(pit, self.shape) 
                ax.add_patch(Rectangle(
                    (c, r),   
                    1, 1,      
                    facecolor='maroon', 
                    edgecolor='red',   
                    lw=2        
                ))
            wandb.log({"Coverage Plot": wandb.Image(fig)})
            self.visited = np.zeros(self.shape)
            plt.close(fig)
        return self.one_hot_obs[state], reward, terminated, truncated, {"prob": prob}

 
    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self._steps = 0
        self.lastaction = None
        self.s = self.categorical_sample(self.init_state, self.np_random)
        # (obs, info) 반환
        return self.one_hot_obs[self.s], {}
    
    def categorical_sample(self, prob_n, np_random):
            """
            prob_n: [0.1, 0.9] 같은 확률 리스트
            반환: 인덱스 (0 또는 1)
            """
            prob_n = np.asarray(prob_n)
            csprob_n = np.cumsum(prob_n)
            return (csprob_n > np_random.random()).argmax()
    
    def _check_bounds(self, coord):
        is_fallen = False
        if coord[0] < 0 or coord[0] >= self.shape[0] or coord[1] < 0 or coord[1] >= self.shape[1]:
            is_fallen = True
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord, is_fallen


class FallEnv(GridCore):

    def __init__(self, pits : list[list[int,int]] = [], **kwargs):
        self.pits = pits
        super(FallEnv, self).__init__(**kwargs)

    def _calculate_transition_prob(self, current : tuple[int,int], delta, prob):
        transitions = []
        for d, p in zip(delta, prob):
            new_position = np.array(current) + np.array(d)
            new_position, is_fallen = self._check_bounds(new_position)
            new_position = new_position.astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            reward = 0
            is_done = False

            if tuple(new_position) == self.goal:
                reward = 1
                is_done = True
            elif is_fallen or (new_state in self.pits):
                reward = -1
                new_state = np.ravel_multi_index(self.start, self.shape)

            transitions.append((p, new_state, reward, is_done))

        return transitions
    
    def _init_transition_probability(self):
        for idx, p in enumerate(self.pits):
            try:
                self.pits[idx] = np.ravel_multi_index(p, self.shape)
            except:
                pass  # <- this has to be here for the agent. Otherwise it throws an unexplainable error
        # Calculate transition probabilities
        P = {}
        for s in range(self.state_size):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.action_size)}
            # other_prob = self.afp / 3.
            tmp = [[UP, DOWN, LEFT, RIGHT],
                   [DOWN, LEFT, RIGHT, UP],
                   [LEFT, RIGHT, UP, DOWN],
                   [RIGHT, UP, DOWN, LEFT]]
            tmp_dirs = [[[-1, 0], [1, 0], [0, -1], [0, 1]],
                        [[1, 0], [0, -1], [0, 1], [-1, 0]],
                        [[0, -1], [0, 1], [-1, 0], [1, 0]],
                        [[0, 1], [-1, 0], [1, 0], [0, -1]]]
            # tmp_pros = [[1 - self.afp, other_prob, other_prob, other_prob],
            #             [1 - self.afp, other_prob, other_prob, other_prob],
            #             [1 - self.afp, other_prob, other_prob, other_prob],
            #             [1 - self.afp, other_prob, other_prob, other_prob], ]
            tmp_pros = [[1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],]
            for acts, dirs, probs in zip(tmp, tmp_dirs, tmp_pros):
                P[s][acts[0]] = self._calculate_transition_prob(position, dirs, probs)
        return P
    