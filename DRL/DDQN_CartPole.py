import random

import gym
import numpy as np
import torch
from torch import nn
import collections
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import ReplayBuffer

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, m):
        return self.net(m)

class DDQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update

        self.count = 0

    def choose_action(self, state):#epsilon greedy
        if np.random.random()<self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        state = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        reward = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_state = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        q_values = self.q_net(state).gather(1, action)
        max_action = self.q_net(next_state).max(1)[1].view(-1,1)
        next_q_values = self.target_q_net(next_state).gather(1, max_action)
        '''
        DDQN与DQN的核心区别就在这里，用q网络选择动作，用target netwrok计算Q值
        '''
        td_target = reward + self.gamma*next_q_values*(1-done)

        loss = F.mse_loss(td_target, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count%self.target_update==0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


lr = 2e-3
gamma = 0.98
epsilon = 0.01
hidden_dim = 128
env = gym.make('CartPole-v0')
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
agent = DDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
num_episodes = 500
return_list = []
for i in range(5):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d'%i)as pbar:#进度条
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size()>minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states':b_s,
                        'actions':b_a,
                        'rewards':b_r,
                        'next_states':b_ns,
                        'dones':b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if(i_episode+1)%10==0:
                pbar.set_postfix({
                    'episode':
                    '%d'%(num_episodes/10*i+i_episode+1),
                    'return':
                    '%.3f'%np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('CartPole-v0'))
plt.show()
