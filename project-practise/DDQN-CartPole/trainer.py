import torch
import torch.nn as nn
from DDQN import DDQN
import gymnasium as gym
from buffer import ReplayBuffer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class trainer:
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.env = gym.make('CartPole-v1')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.agent = DDQN(self.state_dim, self.args.hidden_dim, self.action_dim, self.args.gamma, self.args.epsilon, self.args.target_update,
                 self.args.lr,device)
        self.buffer = ReplayBuffer(self.args.buffer_size)

    def train(self):
        return_list = []
        for i in range(self.args.epoch):
            with tqdm(total=int(self.args.num_episode/10), desc='Iteration %d'%i)as pbar:
                for i_episode in range(int(self.args.num_episode/10)):
                    episode_return = 0
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.agent.choose_action(state)
                        next_state, reward, done, truncated, info = self.env.step(action)
                        self.buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward

                        if self.buffer.__len__() > self.args.minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = self.buffer.sample(self.args.batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'rewards': b_r,
                                'next_states': b_ns,
                                'dones': b_d
                            }
                            self.agent.update(transition_dict)
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.args.num_episode / 10 * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format('CartPole-v0'))
        plt.show()
