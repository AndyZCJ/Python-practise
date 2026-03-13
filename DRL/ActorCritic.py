import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

GAMMA = 0.95#折扣因子
LR = 0.01#学习率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=False #非确定性算法

#策略网络
class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weight(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    def __init__(self, env):
        #状态空间和动作空间
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        #初始化参数
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr =LR)

        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()

        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)

        return action

    def learn(self, state, action, td_error):
        self.time_step += 1

        #前向传播
        softmax_input = self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        #反向传播
        loss_a = -neg_log_prob*td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()

EPSILON = 0.01
REPLAY_SIZE = 10000  # 经验回放buffer大小
BATCH_SIZE = 32
REPLACE_TARGET_FREQ = 10#更新Q网络的频率

#value network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20,1)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weight(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)

class Critic(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.time_step = 0
        self.epsilon = EPSILON

    def train(self, state, reward, next_state):
        s, next_s = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)

        v = self.network.forward(s)
        next_v = self.network.forward(next_s)

        loss_q = self.loss_func(reward+GAMMA*next_v, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward+GAMMA*next_v-v

        return td_error

ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 3000
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        state = env.reset()

        for step in range(STEP):
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            td_error = critic.train(state, reward, next_state)
            actor.learn(state, action, td_error)
            state = next_state
            if done:
                break

        if episode%100==0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')
