import numpy as np

class QLearning:
    def __init__(
        self, 
        n_states, 
        n_actions, 
        lr, 
        discount_factor,
        device
    ):
        self.n_states: int = n_states
        self.n_actions: int = n_actions
        
        self.lr: float = lr
        self.discount_factor: float = discount_factor
        self.device: str = device
        self.q_table = np.zeros(
            (self.n_states, self.n_actions), 
            dtype=np.float32
        )
        self.epsilon: float = 1.0

    def epsilon_decay(self, training_steps, total_training_steps, e_greedy_type: str, power_value: float = 2.0):    
        if e_greedy_type == "inverse":
            self.epsilon = max(0.01, 1.0 / (1.0 + 30 * (training_steps / total_training_steps)))
        elif e_greedy_type == "linear":
            self.epsilon = max(0.01, 1 - training_steps / total_training_steps)
        elif e_greedy_type == "power":
            progress = training_steps / total_training_steps
            self.epsilon = max(0.01, (1.0 - progress) ** power_value)
        else:
            raise ValueError(f"e_greedy_type {e_greedy_type} is not supported.")

    def lr_decay(
        self, 
        training_steps, 
        total_training_steps,
        initial_lr, 
    ):
        final_lr = 0.01 * initial_lr
        progress = training_steps / total_training_steps
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        self.lr = final_lr + (initial_lr - final_lr) * cosine

    def select_action(self, state, deterministic = False):
        if deterministic or np.random.rand() >= self.epsilon:
            # return np.argmax(self.q_table[state]) # 동일한 값이 여러개일 때 가장 작은 인덱스를 반환
            return np.random.choice(
                np.flatnonzero(
                    self.q_table[state] == self.q_table[state].max()
                )
            )
        else:
            return np.random.randint(self.n_actions)
    
    def select_repetition(
        self, 
        state, 
        action, 
        warmup = False, 
        deterministic = False
    ):
        return 1
    
    def update(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done
    ):
        target_q = reward + self.discount_factor * (1 - done) * np.max(self.q_table[next_state])
        td_error = target_q - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        log_dict = {
            "td_error": td_error,
        }
        
        return log_dict