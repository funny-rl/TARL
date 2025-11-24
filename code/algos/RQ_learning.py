import numpy as np
from .Q_learning import QLearning

from typing import Tuple

class RQLearning(QLearning):
    def __init__(
        self, 
        n_states, 
        n_actions, 
        lr, 
        discount_factor,
        max_repetitions,
        env_dict,
        use_rep_max_q,
        alpha,
        device
    ):
        super().__init__(
            n_states, 
            n_actions, 
            lr, 
            discount_factor,
            device
        )
        self.use_rep_max_q: bool = use_rep_max_q
        self.max_repetitions: int = max_repetitions
        self.state_info: list[str] = env_dict["state_info"]
        self.action_info: dict[int, str] = env_dict["action_info"]
        self.reward_info: dict[str, float] = env_dict["reward_info"]
        self.n_rows: int = len(self.state_info)
        self.n_cols: int = len(self.state_info[0]) 
        self.alpha = alpha

    def select_repetition(
        self, 
        state, 
        action, 
        warmup = False, 
        deterministic = False
    ):
        if warmup:
            return np.random.randint(1, self.max_repetitions + 1)
        else:
            next_state_list: list[Tuple[str, int]] = []
            state_row, state_col = divmod(state, self.n_cols)
            action_str = self.action_info[action]
            curr_state = state
            for _ in range(1, self.max_repetitions + 1):
                if action_str == "UP":
                    state_changes = -self.n_cols
                elif action_str == "RIGHT": 
                    state_changes = 1
                elif action_str == "DOWN":
                    state_changes = self.n_cols
                elif action_str == "LEFT":
                    state_changes = -1
                else:
                    raise ValueError("Invalid action.")
                
                next_state = curr_state + state_changes
                # Check for boundaries and obstacles
                
                if next_state < 0 or next_state >= self.n_states:
                    break
                
                next_row, next_col = divmod(next_state, self.n_cols)
                
                if action_str == "UP" or action_str == "DOWN":
                    if state_col != next_col:
                        break
                else:
                    if state_row != next_row:
                        break
                    
                if self.state_info[next_row][next_col] == 'H':
                    next_state_list.append(
                        (
                            self.state_info[next_row][next_col], 
                            next_state
                        )
                    )
                    break
                next_state_list.append(
                    (
                        self.state_info[next_row][next_col], 
                        next_state
                    )
                )
                curr_state = next_state
            
            if len(next_state_list) == 0:
                next_state_list = [
                    (
                        self.state_info[state_row][state_col], 
                        state
                    ) for _ in range(self.max_repetitions)
                ]
            
            if len(next_state_list) < self.max_repetitions: 
                if next_state_list[-1][0] == "S" or next_state_list[-1][0] == "F":
                    next_state_list= next_state_list + [next_state_list[-1]] * (self.max_repetitions - len(next_state_list))
                    assert len(next_state_list) == self.max_repetitions, "Length of next_state_list should be equal to max_repetitions."
                    
            n_step_rep_Qs = [0.0 for _ in range(len(next_state_list))]
            
            for rep_idx in range(len(next_state_list)):
                discounted_reward = 0.0
                next_state = next_state_list[rep_idx][1]
                for discount in range(rep_idx + 1):
                    discounted_reward += self.discount_factor ** discount * self.reward_info[next_state_list[discount][0]]
                if self.use_rep_max_q:
                    n_step_rep_Qs[rep_idx] = discounted_reward + self.discount_factor ** (rep_idx + 1) * np.max(self.q_table[next_state])
                else:
                    n_step_rep_Qs[rep_idx] = discounted_reward + self.discount_factor ** (rep_idx + 1) * (self.alpha * np.max(self.q_table[next_state]) + (1 - self.alpha) * np.mean(self.q_table[next_state]))
            
            
            
            if deterministic or np.random.rand() >= self.epsilon:
                return np.random.choice(
                    np.flatnonzero(
                        n_step_rep_Qs == max(n_step_rep_Qs)
                    )
                ) + 1
            else:
                return np.random.randint(1, self.max_repetitions + 1)