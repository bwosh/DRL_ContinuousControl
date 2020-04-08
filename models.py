import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)     


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x


class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNet, self).__init__()
        self.actor_body = ActorNet(state_size, action_size)
        self.critic_body = CriticNet(state_size, action_size)

        actor_opt = torch.optim.Adam(self.actor_body.parameters(), lr=1e-4)
        critic_opt = torch.optim.Adam(self.critic_body.parameters(), lr=1e-3)

    def forward(self, state):
        action = torch.tanh(self.actor_body(state))
        return action
