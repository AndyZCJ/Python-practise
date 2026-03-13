import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from utils import  ReplayBuffer

'''定义Q网络'''
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


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)#Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)#目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = learning_rate)#优化器
        self.gamma = gamma#折扣因子
        self.epsilon = epsilon#epsilon greedy
        self.target_update = target_update#更新目标网络的频率
        self.count = 0
        self.device = device

    def choose_action(self, state):#epsilon greedy
        if np.random.random()<self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)#当前Q值
        '''
        torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
        定义：从原tensor中获取指定dim和指定index的数据
        torch.gather(0, index)表示输出的结果是行形式
        torch.gather(1, index)表示输出的结果是列形式
        详细解释：https://zhuanlan.zhihu.com/p/352877584
        '''
        print(q_values)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)#下一状态最大Q值
        '''
        形式： torch.max(input) → Tensor
        返回输入tensor中所有元素的最大值
        torch.max(a,0)返回每一列中最大值的那个元素，且返回索引
        torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引
        '''
        print(self.target_q_net(next_states).shape)
        q_targets = rewards + self.gamma*max_next_q_values * (1-dones)#td target
        dqn_loss = F.mse_loss(q_values, q_targets)#损失
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count%self.target_update==0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())#更新目标网络
        self.count += 1

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

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

                if replay_buffer.size()>minimal_size:#buffer的size到达一定大小后才开始训练
                    env.render()
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
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
plt.title('DQN on {}'.format(env_name))
plt.show()