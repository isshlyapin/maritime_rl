import random
import numpy as np
import torch
from torch import nn, optim
from maritime_rl.rl.dueling_ddqn import DuelingDQN
from maritime_rl.rl.replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(self, obs_dim, act_dim, device="cpu",
                 gamma=0.99, lr=5e-4, buffer_size=100000,
                 batch_size=64, tau=0.005):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.net = DuelingDQN(obs_dim, act_dim).to(self.device)
        self.target = DuelingDQN(obs_dim, act_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.act_dim = act_dim
        self.step = 0

    def act(self, state, eps=0.1):
        if random.random() < eps:
            return np.random.uniform(-1,1,2).astype(np.float32)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(s)
        # continuous action: we discretize rudder/throttle here (49 combos optional)
        a = torch.tanh(q[0,:2]).cpu().numpy()  # [-1,1]x[-1,1]
        return a

    def store(self, *args):
        self.buffer.add(*args)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        s,a,r,ns,d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.net(s)
        with torch.no_grad():
            next_q_main = self.net(ns)
            next_q_target = self.target(ns)
            best_actions = next_q_main.argmax(1, keepdim=True)
            target_q = r + self.gamma * (1-d) * next_q_target.gather(1, best_actions)
        loss = nn.MSELoss()(q.max(1, keepdim=True)[0], target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # soft update
        for p, tp in zip(self.net.parameters(), self.target.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
