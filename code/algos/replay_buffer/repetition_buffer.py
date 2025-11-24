import torch
import numpy as np

from .naive_buffer import NaiveReplayBuffer

def np_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        return torch.tensor(x, dtype=torch.float32)

class RepetitionBuffer(NaiveReplayBuffer):
    def __init__(
        self, 
        max_size, 
        state_dim, 
        action_dim, 
        device
    ):
        super(RepetitionBuffer, self).__init__(
            max_size, 
            state_dim, 
            action_dim, 
            device
        )
        self.rep = np.zeros((self.max_size, 1))
        
    def add(
        self, 
        state, 
        action,
        rep, 
        reward, 
        next_state, 
        done, 
    ):
        self.state_buffer[self.ptr] = np_to_torch(state)
        self.action_buffer[self.ptr] = np_to_torch(action)
        self.reward_buffer[self.ptr] = np_to_torch(reward)
        self.next_state_buffer[self.ptr] = np_to_torch(next_state)
        self.done_buffer[self.ptr] = np_to_torch(done)
        self.rep[self.ptr] = np_to_torch(rep)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        return (
            torch.FloatTensor(self.state_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.action_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.rep[idxs]).to(self.device),
            torch.FloatTensor(self.reward_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[idxs]).to(self.device),
            torch.FloatTensor(self.done_buffer[idxs]).to(self.device),
        )
        