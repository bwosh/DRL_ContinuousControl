import numpy as np
import random
import torch

from buffer import ReplayBuffer
from models import ActorCriticNet
from noise import OUNoise
from utils import soft_update

def to_np(t):
    return t.cpu().detach().numpy()

def tensor(t, device):
    return torch.tensor(t, dtype=torch.float).to(device)

class Agent:
    def __init__(self, state_size, action_size, gamma=0.99, 
            update_every = 20*20,  # TODO remove
            update_cycles = 10, # TODO remove
            buffer_size = int(1e5), 
            batch_size=128, 
            tau = 1e-3, 
            device="cuda"):

        print("=== AGENT CREATED ===")
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.update_every = update_every
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_cycles = update_cycles

        self.network = ActorCriticNet(state_size, action_size).to(device)
        self.target_network = ActorCriticNet(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Control variables
        self.noise = OUNoise(action_size, 0)
        self.t_step = 0
        self.memory = ReplayBuffer(action_size,buffer_size,batch_size,0,device)

    def act(self, state, add_noise=True):
        self.network.eval()
        with torch.no_grad():
            action = self.network(tensor(state, self.device).unsqueeze(0))
        action = to_np(action)[0]
        self.network.train()
        if add_noise:
            action += self.noise.sample()
        action = np.clip(action, -1, 1)
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                for uc in range(self.update_cycles):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = tensor(states, self.device)
        actions = tensor(actions, self.device)
        rewards = tensor(rewards, self.device)#.unsqueeze(-1)
        next_states = tensor(next_states, self.device)
        mask = tensor(1 - dones, self.device)#.unsqueeze(-1)

        # TODO check shapes
        #print("states",states.shape)
        #print("actions",actions.shape)
        #print("rewards",rewards.shape)
        #print("next_states",next_states.shape)
        #print("mask",mask.shape)

        a_next = self.target_network.actor(next_states)
        q_targets_next = self.target_network.critic(next_states, a_next)
        q_targets = rewards+ (self.gamma * q_targets_next * mask)

        #print("a_next",a_next.shape)
        #print("q_targets_next",q_targets_next.shape)
        #print("q_targets",q_targets.shape)

        q = self.network.critic(states, actions)
        #print("q",q.shape)
        #exit(0)
        critic_loss = (q - q_targets_next).pow(2).mul(0.5).sum(-1).mean()

        self.network.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.critic.parameters(), 1)
        self.network.critic_opt.step()

        action = self.network.actor(states)
        policy_loss = -self.network.critic(states.detach(), action).mean()

        self.network.actor_opt.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        soft_update(self.network, self.target_network, self.tau)

    def save(self):
        torch.save(self.network.state_dict(),"net.pth")
        torch.save(self.target_network.state_dict(),"target.pth")
