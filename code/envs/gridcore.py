import numpy as np
import gymnasium as gym
from gymnasium import spaces


LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

class GridCore(gym.Env):
    metadata = {'render.modes': ["rgb_array"]}

    def __init__(
        self, 
        shape: tuple[int] = (6, 10),
        start: tuple[int] = (0, 0),
        goal: tuple[int] = (0, 9),
        max_steps: int = 1000
    ):
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

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self._steps = 0
        self.lastaction = None
        self.s = self.categorical_sample(self.init_state, self.np_random)
        # (obs, info) 반환
        return self.one_hot_obs[self.s], {}

    def categorical_sample(self, prob_n, np_random):
            """
            prob_n: ex. [0.1, 0.9] 
            return index (0 or 1)
            """
            prob_n = np.asarray(prob_n)
            csprob_n = np.cumsum(prob_n)
            return (csprob_n > np_random.random()).argmax()

    def step(self, action):
        action = int(action)
        self._steps += 1

        transitions = self.P[self.s][action]
        i = self.categorical_sample([t[0] for t in transitions], self.np_random)
        prob, state, reward, done = transitions[i]

        self.prev_s = self.s
        self.s = state
        self.lastaction = action

        terminated = done  
        truncated = False
        
        if self._steps >= self.max_steps:
            truncated = True
        return self.one_hot_obs[state], reward, terminated, truncated, {"prob": prob}

    def _check_bounds(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    
    def render(self, mode="rgb_array"):
        """
        Render grid as RGB array with specific color mappings and grid lines.
        
        Modifications:
        1. Start (self.start) is colored Green.
        2. Pits/Cliff (X) are colored Red.
        3. Goal (G) is colored Blue.
        4. Thin gray lines are drawn to separate grid cells.
        5. Agent is marked by a black circle (using self.prev_s).
        """

        if mode != "rgb_array":
            raise NotImplementedError("Only rgb_array mode is supported.")

        # grid info
        H, W = self.shape
        cell_size = 40
        img = np.zeros((H * cell_size, W * cell_size, 3), dtype=np.uint8)

        # colors
        color_map = {
            ".": (255, 255, 255),    
            "S": (0, 150, 0),        
            "G": (0, 0, 200),        
            "X": (200, 0, 0),        
        }
        grid_line_color = (200, 200, 200)
        line_thickness = 1

        # build grid
        grid = np.full(self.shape, ".", dtype=object)

        sx, sy = self.start
        grid[sx, sy] = "S"

        pits = getattr(self, "pits", [])
        for p in pits:
            pit_coord = np.unravel_index(p, self.shape)
            grid[pit_coord] = "X"

        gx, gy = self.goal
        grid[gx, gy] = "G"

        ax, ay = np.unravel_index(self.prev_s, self.shape) 

        for i in range(H):
            for j in range(W):
                cell_type = grid[i, j]
                c = color_map[cell_type]
                
                # 셀 영역 칠하기
                img[i*cell_size:(i+1)*cell_size,
                    j*cell_size:(j+1)*cell_size] = c

        for i in range(1, H):
            start_y = i * cell_size
            img[start_y - line_thickness//2 : start_y + line_thickness//2 + 1, :] = grid_line_color
        
        # 수직선 그리기
        for j in range(1, W):
            start_x = j * cell_size
            img[:, start_x - line_thickness//2 : start_x + line_thickness//2 + 1] = grid_line_color

        cx = ax * cell_size + cell_size // 2    # center x
        cy = ay * cell_size + cell_size // 2    # center y
        radius = cell_size // 4                 # radius of circle

        # draw circle using (x - cx)^2 + (y - cy)^2 < r^2
        Y, X = np.ogrid[:H * cell_size, :W * cell_size]
        dist2 = (X - cy)**2 + (Y - cx)**2
        mask = dist2 <= radius**2

        img[mask] = (0, 0, 0)  # black circle

        return img

class FallEnv(GridCore):

    def __init__(
        self, 
        pits : list[list[int,int]] = [], 
        **kwargs
    ):
        self.pits = pits
        super(FallEnv, self).__init__(**kwargs)

    def _calculate_transition_prob(self, current : tuple[int,int], delta, prob):
        transitions = []
        for d, p in zip(delta, prob):
            new_position = np.array(current) + np.array(d)
            new_position = self._check_bounds(new_position)
            new_position = new_position.astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            reward = -1
            is_done = False
            if new_position.tolist() == self.goal:
                is_done = True
            elif new_state in self.pits:
                reward = -100
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