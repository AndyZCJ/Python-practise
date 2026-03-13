import collections
import random
import numpy as np
class ReplayBuffer:#经验回放的buffer
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)#使用deque作为buffer

    def add(self, state, action, reward, next_state, done):#放入transition
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):#随机抽取transition，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        '''
        zip用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        语法：
        zip([iterable, ...])
        '''
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
