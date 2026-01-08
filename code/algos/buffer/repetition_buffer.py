import torch

class SkipBuffer:
    def __init__(
        self, 
        buffer_size, 
        state_dim, 
        action_dim, 
        gamma,
        prev_buffer_save,
        device
    ):
        self.buffer_size: int = buffer_size
        self.device: str = device
        self.gamma: float = gamma
        self.prev_buffer_save: bool = prev_buffer_save

        self.ptr: int = 0
        self.size: int = 0
        if isinstance(state_dim, int):
            state_dim = (state_dim,)
            
        self.state_buffer = torch.zeros((buffer_size, *state_dim))
        self.action_buffer = torch.zeros((buffer_size, action_dim)) 
        self.repetition_buffer = torch.zeros((buffer_size, 1))
        self.reward_buffer = torch.zeros((buffer_size, 1))
        self.next_state_buffer = torch.zeros((buffer_size, *state_dim))
        self.not_done = torch.zeros((buffer_size, 1))
    
    def add(
        self,
        state,
        action,
        repetition,
        reward,
        next_state,
        done
    ):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action if torch.is_tensor(action) else torch.tensor(action)
        self.repetition_buffer[self.ptr] = repetition
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        return (
            torch.FloatTensor(self.state_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.action_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.repetition_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.reward_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.not_done[idxs]).to(self.device),
        )
        
    def transform(
        self,
        skip_states,
        action,
        skip_rewards,
        skip_dones,
        next_skip_states,
        repetition,
        buffer
    ):
        if self.prev_buffer_save:
            for n_idx, end_state in enumerate(next_skip_states):
                next_idx = n_idx + 1
                for idx, start_state in enumerate(skip_states[:next_idx]):
                    skip_reward = 0
                    skip_step = 0
                    for exp, r in enumerate(skip_rewards[idx : next_idx]):
                        skip_reward +=  self.gamma ** exp * r
                        skip_step += 1
                    
                    buffer.add(
                        state = start_state,
                        action = action,
                        repetition = skip_step,
                        reward = skip_reward,
                        next_state = end_state,
                        done = skip_dones[n_idx]
                    )
                    
        else:
            pad_len = max(0, repetition - len(skip_states))
            real_rep = len(skip_states)
            if pad_len > 0:
                    empty_state = torch.zeros_like(skip_states[0])
                    skip_states.extend([empty_state] * pad_len)
                    skip_rewards.extend([0.0] * pad_len)
                    skip_dones.extend([True] * pad_len)
                    next_skip_states.extend([empty_state] * pad_len)

            for n_idx, end_state in enumerate(next_skip_states):
                next_idx = n_idx + 1
                rep_states = min(real_rep, next_idx)
                for idx, start_state in enumerate(skip_states[:rep_states]):
                    skip_reward = 0
                    skip_step = 0
                    for exp, r in enumerate(skip_rewards[idx : next_idx]):
                        skip_reward +=  self.gamma ** exp * r
                        skip_step += 1
                    
                    buffer.add(
                        state = start_state,
                        action = action,
                        repetition = skip_step,
                        reward = skip_reward,
                        next_state = end_state,
                        done = skip_dones[n_idx]
                    )
                
        return buffer