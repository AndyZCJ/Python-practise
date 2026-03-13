import collections
import random
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        super().__init__()
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transition = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transition)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)