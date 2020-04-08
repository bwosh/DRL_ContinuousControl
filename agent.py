import numpy as np
import random

from buffer import ReplayBuffer
from models import ActorCriticNet

class Agent:
    def __init__(self, state_size, action_size, buffer_size = 2**10, batch_size=4, device="cpu"):
        print("=== AGENT CREATED ===")
        self.state_size = state_size
        self.action_size = action_size

        self.network = ActorCriticNet(state_size, action_size)
        self.target_network = ActorCriticNet(state_size, action_size)
        self.target_network.load_state_dict(self.network.state_dict())

        self.replay = ReplayBuffer(action_size,buffer_size,batch_size,0,device)

    def act(self, state):
        return np.random.random(self.action_size)

    def step(self, state, action, reward, next_state, done):
        pass
