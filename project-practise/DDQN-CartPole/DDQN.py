import torch
import torch.nn as nn
import torch.nn.functional as F
from model import QNet
import numpy as np
from buffer import ReplayBuffer
class DDQN:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, epsilon, target_update,
                 learning_rate,device):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.Q_Network = QNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network = QNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q_Network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.buffer = ReplayBuffer(10000)
        self.count = 0

    def choose_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
        if np.random.random() > self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            action = self.Q_Network(state).argmax().item()
            return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        done = torch.tensor(transition_dict['dones']).view(-1, 1).to(torch.float).to(self.device)

        q_value = self.Q_Network(states).gather(1, actions)
        max_action = self.Q_Network(next_states).max(1)[1].view(-1, 1)
        next_q_value = self.target_network(next_states).gather(1, actions)

        td_target = rewards + self.gamma * next_q_value*(1-done)
        td_error = F.mse_loss(q_value, td_target)
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        if self.count % self.target_update==0:
            self.target_network.load_state_dict(self.Q_Network.state_dict())

        self.count += 1

