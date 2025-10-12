import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,128,64)):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.value = nn.Linear(hidden[2], 1)
        self.adv = nn.Linear(hidden[2], act_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        v = self.value(x)
        a = self.adv(x)
        return v + a - a.mean(1, keepdim=True)
