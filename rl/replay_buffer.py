import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def add(self, s,a,r,ns,d):
        self.buffer.append((s,a,r,ns,d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = map(np.array, zip(*batch))
        return s,a,r,ns,d
    def __len__(self):
        return len(self.buffer)
