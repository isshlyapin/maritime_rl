import torch
import random

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from ddqn_model import DQN

class MaritimeDoubleDQNAgent:
    def __init__(self, 
            state_dim, 
            action_dim,
            lr=0.0005, 
            gamma=0.99, 
            buffer_size=100_000, 
            batch_size=64,
            target_update=1_000, 
            weights = None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        if weights is not None:
            self.policy_net.load_state_dict(torch.load(weights))

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training counters
        self.steps_done = 0

    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy policy with Double DQN"""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, filepath):
        """Save the model weights to a file"""
        torch.save(self.target_net.state_dict(), filepath)

    def update_model(self):
        """Update the model using Double DQN"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN: use policy net to select actions, target net to evaluate
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).gather(1, next_actions)
        
        # Target Q values
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()        
        self.optimizer.step()
        
        # Update target network
        self.steps_done += self.batch_size
        if self.steps_done > self.target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_update *= 2