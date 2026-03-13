import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gym
import numpy as np
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from collections import deque
LR = 0.01  # 学习率
GAMMA = 0.9         # 奖励的折扣因子
EPSILON = 0.9       # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER = 100                 # 目标网络更新的频率
env = gym.make('CartPole-v0')             # 加载游戏环境
N_ACTIONS = env.action_space.n            # 动作数
N_SPACES = env.observation_space.shape[0] # 状态数量
env = env.unwrapped
env.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def init_weights(m) :#初始化参数
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)

class Q_Network(nn.Module):#Q网络
    def __init__(self):
        super(Q_Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SPACES, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS)
        )

    def forward(self, m):
        output = self.net(m)
        return output

class DQN:
    def __init__(self):
        self.net, self.target_net = Q_Network().apply(init_weights),Q_Network().apply(init_weights)
        self.net.to(device)
        self.target_net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.loss_function = nn.MSELoss()
        self.memory = deque(maxlen=1000000)
        self.learn_every = 3
        self.curr_step = 0
        self.batch_size = 32
        self.burnin = 1000
        self.target_update_freq = 100
        self.memory_counter = 0

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s).cuda(), dim=0)
        if np.random.uniform()<EPSILON:
            action_value = self.net(s)
            action = action_value.argmax().item()
        else:
            action = np.random.randint(0, N_ACTIONS)

        return action

    def learn(self):
        if self.memory_counter<self.burnin:
            return
        s, a, r, s_,done = self.recall()
        s = torch.FloatTensor(s).to(device)
        s_ = torch.FloatTensor(s_).to(device)
        a = torch.tensor(a).view(-1,1).to(device)
        r = torch.FloatTensor(r).view(-1, 1).to(device)

        action_value = self.net(s)
        current_q = action_value.gather(1, a)
        q_ = self.target_net(s_).max(1)[0].view(-1, 1)
        td_target = r + GAMMA * q_

        loss = self.loss_function(current_q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.curr_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.curr_step += 1

    def cache(self, s, a, r, s_, done):

        self.memory.append((s, a, r, s_, done,))
        self.memory_counter += 1

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)
        return state, next_state, action, reward, done





DQN = DQN()
x_axis, y_axis = [], []

for epoch in range(1000):
    s = env.reset()
    ep_r = 0
    while True:
        #env.render()
        a = DQN.choose_action(s)

        s_, r, done, info = env.step(a)
        DQN.cache(s, a, r, s_, done)
        ep_r+=r

        DQN.learn()
        if done:
            break

        s = s_
    print(f'Ep: {epoch} | Ep_r: {round(ep_r, 2)}')
    x_axis.append(epoch)
    y_axis.append(ep_r)
plt.plot(x_axis, y_axis)
plt.show()
