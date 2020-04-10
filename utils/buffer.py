import random
import numpy as np
import torch

from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def float_tensor(self, data):
        return torch.from_numpy(np.vstack(data)).float().to(self.device)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.float_tensor([e.state for e in experiences if e is not None])
        actions = self.float_tensor([e.action for e in experiences if e is not None])
        rewards = self.float_tensor([e.reward for e in experiences if e is not None])
        next_states = self.float_tensor([e.next_state for e in experiences if e is not None])
        dones = self.float_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)