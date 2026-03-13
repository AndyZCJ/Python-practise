import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

LR_ACTOR = 0.0001     # 策略网络的学习率
LR_CRITIC = 0.0001   # 价值网络的学习率
GAMMA = 0.99         # 奖励的折扣因子
EPSILON = 0.9       # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER = 10                 # 目标网络更新的频率
env = gym.make('CartPole-v0')             # 加载游戏环境
N_ACTIONS = env.action_space.n            # 动作数
N_SPACES = env.observation_space.shape[0] # 状态数量
env = env.unwrapped
env.seed(0)
torch.manual_seed(0)
# 网络参数初始化，采用均值为 0，方差为 0.1 的高斯分布
def init_weights(m) :
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)


class AC_Network(nn.Module):#actor网络
    def __init__(self):
        super(AC_Network, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(N_SPACES, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, N_ACTIONS)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(N_SPACES, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, s):
        pi = self.actor_net(s)
        pi = F.softmax(pi, dim=-1)
        v = self.critic_net(s)
        return pi, v



class A2C:
    def __init__(self):
        self.net = AC_Network().apply(init_weights)#初始化
        self.actor_optimizer = optim.Adam(self.net.actor_net.parameters(), lr=LR_ACTOR)#actor优化器
        self.critic_optimizer = optim.Adam(self.net.critic_net.parameters(), lr=LR_CRITIC)
        self.loss_function = nn.MSELoss()#损失函数

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), dim=0)
        if np.random.uniform()<EPSILON:
            action_value, _ = self.net(s)
            action = torch.max(action_value, dim=1)[1].item()
        else:
            action = np.random.randint(0,N_ACTIONS)

        return action

    def learn(self, s, a, r, s_):
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)

        pi, v = self.net(s)
        pi_, v_ = self.net(s_)
        v_.detach()
        target = r + GAMMA*v_
        advantage = (v-target).detach()
        log_q_actor = torch.log(pi)
        actor_loss = log_q_actor[a]*advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        loss_critic = self.loss_function(v,target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

a2c = A2C()
x, y = [], []
for epoch in range(2000) :
    s = env.reset()
    ep_r = 0
    while True :
        env.render()
        a = a2c.choose_action(s)       # 选择动作

        s_, r, done, info = env.step(a)# 执行动作


        ep_r += r

        # 学习
        a2c.learn(s, a, r, s_)

        if done :
            break

        s = s_
    print(f'Ep: {epoch} | Ep_r: {round(ep_r, 2)}')
    x.append(epoch)
    y.append(ep_r)
plt.plot(x, y)
plt.show()