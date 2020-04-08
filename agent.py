import numpy as np
import random
import torch

from buffer import ReplayBuffer
from models import ActorCriticNet

def to_np(t):
    return t.cpu().detach().numpy()

def from_np(t, device):
    return torch.tensor(t, dtype=torch.float).to(device)

class Agent:
    def __init__(self, state_size, action_size, buffer_size = 2**10, batch_size=4, device="cpu"):
        print("=== AGENT CREATED ===")
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.network = ActorCriticNet(state_size, action_size)
        self.target_network = ActorCriticNet(state_size, action_size)
        self.target_network.load_state_dict(self.network.state_dict())

        self.replay = ReplayBuffer(action_size,buffer_size,batch_size,0,device)

    def act(self, state, eps):
            if np.random.random() < eps:
                random_action = np.random.random(self.action_size)
                return random_action
            else:
                action = self.network(from_np(state, self.device))
                action = to_np(action)
                return action

    def step(self, state, action, reward, next_state, done):
        pass
