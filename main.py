import math
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Value stream
        self.value_stream = nn.Linear(64, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class MultiShipCollisionAvoidance:
    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.99, 
                 buffer_size=100_000, batch_size=64, target_update=1_000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
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
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_model(self, verbose=False):
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
        
        # Calculate gradient norm for debugging
        total_norm = 0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if verbose:
                print(f"  [DEBUG] Target network updated at step {self.steps_done}")
        
        # Return debug info
        debug_info = {
            'loss': loss.item(),
            'grad_norm': total_norm,
            'q_mean': current_q_values.mean().item(),
            'q_std': current_q_values.std().item(),
            'q_max': current_q_values.max().item(),
            'q_min': current_q_values.min().item(),
            'target_q_mean': target_q_values.mean().item(),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
        }
        
        return debug_info


class MaritimeEnvironment:
    def __init__(self, num_ships=5, k_nearest=5):
        self.num_ships = num_ships
        self.k_nearest = k_nearest
        self.time_step = 0
        self.max_steps = 2000

        # Map parameters
        self.size = 500 # meters
        
        # Ship parameters
        self.max_speed = 10           # meters
        self.max_heading_change = 30  # degrees
        
        # Safety parameters
        self.d_safe_min = 10     # meters
        self.d_safe_max = 100    # meters
        
        # State dimensions
        self.state_dim = 4 + k_nearest * 3  # own ship + k nearest ships (dist, rel_speed, rel_heading)
        
        # Action space discretization
        self.speed_changes = [-3, -2, -1, 0, 1, 2, 3]  # knots
        self.heading_changes = [-30, -20, -10, 0, 10, 20, 30]  # degrees
        self.action_dim = len(self.speed_changes) * len(self.heading_changes)
        
        self.reset()
    
    def reset(self, verbose=False):
        """Reset environment to initial state"""
        self.time_step = 0
        
        # Initialize ships with random positions, speeds, and headings
        self.ships = []
        for i in range(self.num_ships):
            ship = {
                'x': random.uniform(0, self.size),
                'y': random.uniform(0, self.size),
                'target_x': random.uniform(0, self.size),
                'target_y': random.uniform(0, self.size),
                'speed': random.uniform(0, self.max_speed),
                'heading': random.uniform(-180, 180),
                'desired_speed': 7
            }
            self.ships.append(ship)
        
        if verbose:
            ship = self.ships[0]
            dist_to_target = math.sqrt((ship['target_x'] - ship['x'])**2 + (ship['target_y'] - ship['y'])**2)
            print(f"  [RESET] Initial distance to target: {dist_to_target:.1f}m")
            print(f"  [RESET] Initial position: ({ship['x']:.1f}, {ship['y']:.1f})")
            print(f"  [RESET] Target position: ({ship['target_x']:.1f}, {ship['target_y']:.1f})")
        
        return self._get_state(0, verbose=verbose)  # Return state for ship 0
    
    def _get_k_nearest_ships(self, ship_idx):
        """Get K nearest ships to the given ship"""
        ship = self.ships[ship_idx]
        distances = []
        
        for i, other_ship in enumerate(self.ships):
            if i == ship_idx:
                continue
                
            dx = other_ship['x'] - ship['x']
            dy = other_ship['y'] - ship['y']
            distance = math.sqrt(dx**2 + dy**2)
            distances.append((i, distance))
        
        # Sort by distance and take K nearest
        distances.sort(key=lambda x: x[1])
        return [idx for idx, dist in distances[:self.k_nearest]]
    
    def _get_state(self, ship_idx, verbose=False):
        """Get state representation for a ship"""
        ship = self.ships[ship_idx]
        state = [
            (ship['target_x'] - ship['x']) / self.size, # normalized 
            (ship['target_y'] - ship['y']) / self.size, # normalized
            ship['speed'] / self.max_speed,             # normalized
            ship['heading'] / 180.0                     # normalized
        ]
        
        # Add information about K nearest ships
        nearest_ships = self._get_k_nearest_ships(ship_idx)
        
        for nearest_idx in nearest_ships:
            other_ship = self.ships[nearest_idx]
            
            # Relative distance (normalized)
            dx = other_ship['x'] - ship['x']
            dy = other_ship['y'] - ship['y']
            distance = math.sqrt(dx**2 + dy**2) / self.size
            
            # Relative speed
            rel_speed = abs(self._relative_velocity(other_ship, ship)) / self.max_speed
            
            # Relative heading (angle difference)
            heading_diff = self._heading_diff(other_ship, ship) / 180.0
            
            state.extend([distance, rel_speed, heading_diff])
        
        # Pad with zeros if fewer than K nearest ships
        while len(state) < self.state_dim:
            state.extend([0, 0, 0])
        
        state_array = np.array(state, dtype=np.float32)
        
        if verbose:
            print(f"  [STATE] Raw state (first 7 dims): {state_array[:7]}")
            print(f"  [STATE] State range: [{state_array.min():.3f}, {state_array.max():.3f}]")
            print(f"  [STATE] State mean: {state_array.mean():.3f}, std: {state_array.std():.3f}")
        
        return state_array

    def _apply_ship_motion(self, ship, delta_speed, delta_heading, dt=1.0):
        """Применить изменения скорости и курса и обновить позицию"""
        ship['speed'] = max(0.0, min(self.max_speed, ship['speed'] + delta_speed))
        ship['heading'] = self._normalize_angle_180(ship['heading'] + delta_heading)

        ship['x'] += ship['speed'] * math.cos(math.radians(ship['heading'])) * dt
        ship['y'] += ship['speed'] * math.sin(math.radians(ship['heading'])) * dt

    def _calculate_collision_risk(self, ship_idx):
        """Calculate collision risk for a ship"""
        ship = self.ships[ship_idx]
        max_risk = 0
        
        for i, other_ship in enumerate(self.ships):
            if i == ship_idx:
                continue
                
            # Calculate distance
            dx = other_ship['x'] - ship['x']
            dy = other_ship['y'] - ship['y']
            distance = math.sqrt(dx**2 + dy**2)
            
            # Calculate relative speed
            rel_speed = abs(self._relative_velocity(other_ship, ship))
            
            # Collision risk formula from paper
            if distance <= self.d_safe_min:
                risk = -1.0
            elif self.d_safe_min < distance < self.d_safe_max:
                risk = (distance - self.d_safe_max) / (self.d_safe_max - self.d_safe_min) * (rel_speed / self.max_speed)
            else:
                risk = 0.0
            
            max_risk = min(max_risk, risk)
        
        return max_risk
    
    def _calculate_reward(self, ship_idx, action, verbose=False):
        """Calculate reward for a ship's action"""
        ship = self.ships[ship_idx]
        
        # Collision avoidance reward
        r_ca = self._calculate_collision_risk(ship_idx)
        
        # Navigation efficiency reward
        dx = ship['target_x'] - ship['x']
        dy = ship['target_y'] - ship['y']
        desired_heading = math.degrees(math.atan2(dy, dx))

        speed_dev = abs(ship['speed'] - ship['desired_speed']) / self.max_speed
        heading_dev = abs(self._normalize_angle_180(ship['heading'] - desired_heading)) / 180.0
        r_ne = - (speed_dev + heading_dev) / 2
        
        # COLREGs compliance reward (simplified)
        r_ce = self._check_colregs_compliance(ship_idx)
        
        # Combined reward with weights from paper
        alpha, beta, gamma = 0.4, 0.4, 0.2
        total_reward = alpha * r_ca + beta * r_ne + gamma * r_ce
        
        if verbose:
            dist_to_target = math.sqrt(dx**2 + dy**2)
            print(f"  [REWARD] r_ca={r_ca:.3f}, r_ne={r_ne:.3f}, r_ce={r_ce:.3f}, total={total_reward:.3f}")
            print(f"  [STATE] speed={ship['speed']:.2f}, heading={ship['heading']:.1f}°, dist_to_target={dist_to_target:.1f}m")
            print(f"  [DEVIATIONS] speed_dev={speed_dev:.3f}, heading_dev={heading_dev:.3f}")
        
        return total_reward
    
    #TODO Ошибка реализации правил!!!
    def _check_colregs_compliance(self, ship_idx):
        """Simplified COLREGs compliance check"""
        # This is a simplified implementation
        # In practice, this would be more complex based on relative bearings, etc.
        # ship = self.ships[ship_idx]
        
        # for i, other_ship in enumerate(self.ships):
        #     if i == ship_idx:
        #         continue
                
        #     # Calculate relative bearing
        #     dx = other_ship['x'] - ship['x']
        #     dy = other_ship['y'] - ship['y']
        #     relative_bearing = math.degrees(math.atan2(dy, dx)) % 360
            
        #     # Simplified rule: avoid crossing from right
        #     if 0 <= relative_bearing <= 90:
        #         return 1.0  # Good compliance
        #     elif 270 <= relative_bearing <= 360:
        #         return 0.5  # Moderate compliance
        #     else:
        #         return 0.0  # Poor compliance
        
        return 0.0  # Default good compliance if no other ships
    
    def step(self, ship_idx, action, verbose=False):
        """Execute action for a ship and return next state, reward, done"""
        dt = 1.0  # time step in seconds

        for idx, ship in enumerate(self.ships):
            if idx == ship_idx:
                speed_idx = action // len(self.heading_changes)
                heading_idx = action % len(self.heading_changes)

                delta_speed = self.speed_changes[speed_idx]
                delta_heading = self.heading_changes[heading_idx]
            else:
                delta_speed = 0.0
                delta_heading = 0.0

            self._apply_ship_motion(ship, delta_speed, delta_heading, dt)
        
        # Calculate reward
        reward = self._calculate_reward(ship_idx, action, verbose=verbose)
        
        # Check if episode is done
        self.time_step += 1
        done = self.time_step >= self.max_steps
        termination_reason = None
        
        # Check for collisions
        if self._calculate_collision_risk(ship_idx) < (-1 + 1e-6):
            reward = -5
            done = True
            termination_reason = "collision"
        
        # Check achieving a goal
        ship = self.ships[ship_idx]
        target_d = math.sqrt(
            (ship['x'] - ship['target_x'])**2 +
            (ship['y'] - ship['target_y'])**2
        )
        if target_d < 50:
            reward = 5
            done = True
            termination_reason = "goal_reached"
        
        if done and termination_reason is None:
            termination_reason = "max_steps"

        next_state = self._get_state(ship_idx, verbose=verbose)
        
        if verbose:
            print(f"  [STEP] Action: speed_delta={delta_speed}, heading_delta={delta_heading}, reward={reward:.3f}, done={done}, reason={termination_reason}")
        
        return next_state, reward, done, termination_reason
    
    def _normalize_angle_180(self, angle_deg):
        """Приводит угол к диапазону [-180, 180]"""
        return (angle_deg + 180) % 360 - 180
    
    def _heading_diff(self, ship1, ship2):
        heading_diff = ship1['heading'] - ship2['heading']
        heading_diff = self._normalize_angle_180(heading_diff)
        return heading_diff

    def _relative_velocity(self, ship1, ship2):
        """Calculate relative velocity between two ships"""
        
        # Convert to radians
        h1 = math.radians(ship1['heading'])
        h2 = math.radians(ship2['heading'])
        
        # Calculate velocity vectors
        v1x = ship1['speed'] * math.cos(h1)
        v1y = ship1['speed'] * math.sin(h1)
        
        v2x = ship2['speed'] * math.cos(h2) 
        v2y = ship2['speed'] * math.sin(h2)
        
        # Relative velocity = difference of velocity vectors
        rel_v = math.sqrt((v1x - v2x)**2 + (v1y - v2y)**2)
        
        return rel_v


def train_model():
    """Main training function"""
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Environment parameters
    num_ships = 25
    k_nearest = 5
    
    # Create environment and agent
    env = MaritimeEnvironment(num_ships=num_ships, k_nearest=k_nearest)
    agent = MultiShipCollisionAvoidance(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=0.0005,
        gamma=0.99,
        buffer_size=100000,
        batch_size=1024,
        target_update=1000
    )
    
    print(f"Environment setup:")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print(f"  Number of ships: {num_ships}")
    print(f"  K nearest ships: {k_nearest}")
    print()
    
    # Training parameters
    num_episodes = 10_000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 500_000
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay
    
    # Training statistics
    episode_rewards = []
    collision_rates = []
    
    print("Starting training...")
    
    epsilon = epsilon_start
    episode_collisions = 0
    
    # Debug tracking
    losses = []
    q_values_log = []
    grad_norms = []
    
    for episode in range(num_episodes):
        # Verbose logging for early episodes
        verbose = (episode < 5 or episode % 100 == 0)
        
        # Reset environment
        state = env.reset(verbose=verbose)
        total_reward = 0
        
        step_count = 0
        done = False
        episode_actions = []
        episode_rewards_list = []
        termination_reason = None

        while not done and step_count < env.max_steps:
            # Select and execute action
            action = agent.select_action(state, epsilon)
            episode_actions.append(action)
            
            step_verbose = verbose and step_count < 10  # Only first 10 steps
            next_state, reward, done, termination_reason = env.step(0, action, verbose=step_verbose)
            episode_rewards_list.append(reward)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.memory) < agent.batch_size:
                continue

            # Update model
            debug_info = agent.update_model(verbose=step_verbose)
            
            if debug_info is not None:
                losses.append(debug_info['loss'])
                grad_norms.append(debug_info['grad_norm'])
                q_values_log.append(debug_info['q_mean'])
            
            # Update statistics
            total_reward += reward
            if reward < -5 + 1e-3:  # Collision penalty
                episode_collisions += 1
    
            # Calculate current epsilon
            epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
            
            state = next_state
            step_count += 1
        
        # Record statistics
        episode_rewards.append(total_reward)
        collision_rate = episode_collisions / (episode + 1)
        collision_rates.append(collision_rate)
        
        # Print progress
        if episode % 1 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_collision = np.mean(collision_rates[-100:])
            
            # Additional debug info
            buffer_size = len(agent.memory)
            recent_loss = np.mean(losses[-100:]) if losses else 0
            recent_grad_norm = np.mean(grad_norms[-100:]) if grad_norms else 0
            recent_q = np.mean(q_values_log[-100:]) if q_values_log else 0
            
            print(f"Episode {episode}/{num_episodes} | Steps: {step_count} | Reason: {termination_reason}")
            print(f"  Reward: {total_reward:.2f} (avg: {avg_reward:.2f}) | Epsilon: {epsilon:.3f}")
            print(f"  Loss: {recent_loss:.4f} | Grad: {recent_grad_norm:.4f} | Q-mean: {recent_q:.4f}")
            print(f"  Buffer: {buffer_size}/{agent.memory.maxlen} | Collisions: {avg_collision:.3f}")
            
            if verbose:
                action_distribution = np.bincount(episode_actions, minlength=env.action_dim)
                most_common_actions = np.argsort(action_distribution)[-3:][::-1]
                print(f"  [DEBUG] Most common actions: {most_common_actions}, counts: {action_distribution[most_common_actions]}")
                print(f"  [DEBUG] Reward distribution: min={min(episode_rewards_list):.3f}, "
                      f"max={max(episode_rewards_list):.3f}, mean={np.mean(episode_rewards_list):.3f}")
            print()
            
        # Save progress
        if episode % 10 == 0:
            name = "models/ship_collision_avoidance_model" + str(episode) + ".pth"
            torch.save(agent.target_net.state_dict(), name)

    
    print("Training completed!")
    return agent, episode_rewards, collision_rates


def test_model(agent, env, num_episodes=100):
    """Test the trained model"""
    print("Testing model...")
    
    total_rewards = []
    collision_counts = []
    goal_reached_counts = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        collisions = 0
        done = False
        goal_reached = False
        
        while not done:
            action = agent.select_action(state, epsilon=0.05)  # Small epsilon for testing
            next_state, reward, done, termination_reason = env.step(0, action)
            
            total_reward += reward
            if reward <= -5:  # Collision
                collisions += 1
            if termination_reason == "goal_reached":
                goal_reached = True
            
            state = next_state
        
        total_rewards.append(total_reward)
        collision_counts.append(collisions)
        goal_reached_counts.append(1 if goal_reached else 0)
    
    avg_reward = np.mean(total_rewards)
    avg_collisions = np.mean(collision_counts)
    goal_reached_rate = np.mean(goal_reached_counts)
    success_rate = (num_episodes - np.sum(collision_counts)) / num_episodes
    
    print(f"Test Results:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Collisions: {avg_collisions:.2f}")
    print(f"  Goal Reached Rate: {goal_reached_rate:.3f}")
    print(f"  Success Rate (no collision): {success_rate:.3f}")
    
    return total_rewards, collision_counts


if __name__ == "__main__":
    # Train the model
    trained_agent, rewards, collisions = train_model()
    
    # Test the model
    test_env = MaritimeEnvironment(num_ships=500, k_nearest=5)
    test_rewards, test_collisions = test_model(trained_agent, test_env)