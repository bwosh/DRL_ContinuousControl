import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=4, fc2_units=8):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) 


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=4, fc2_units=8):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNet, self).__init__()
        self.actor = ActorNet(state_size, action_size)
        self.critic = CriticNet(state_size, action_size)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def forward(self, state):
        action = self.actor(state)
        return action
