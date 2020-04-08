import numpy as np
import random

class Agent:
    def __init__(self, state_size, action_size):
        print("=== AGENT CREATED ===")
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        return np.random.random(self.action_size)

    def step(self, state, action, reward, next_state, done):
        pass
