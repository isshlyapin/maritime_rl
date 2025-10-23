import os

import numpy as np

from ddqn_agent import MaritimeDoubleDQNAgent
from rl_wrapper_maritime_env import RLWrapperMaritimeEnv
import rl_config as cfg

def train_model():
    """Main training function"""
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Environment parameters
    map_size = cfg.MAP_SIZE
    max_steps = int(cfg.MAX_STEPS)
    k_nearest = int(cfg.NUMBER_OF_NEAREST_SHIPS)
    num_ships = int(cfg.NUMBER_OF_SHIPS)
    controlled_ship_idx = 0
    
    # Create environment wrapper
    env = RLWrapperMaritimeEnv(
        map_size=map_size,
        max_steps=max_steps,
        k_nearest=k_nearest,
        num_ships=num_ships,
        ships=None,  # Will generate random ships
        test_idx=controlled_ship_idx
    )
    
    # Calculate state and action dimensions
    # State: [dx, dy, speed, heading] + k_nearest * [distance, rel_speed, rel_heading]
    state_dim = 4 + k_nearest * 3
    # Action: speed_changes (7) * heading_changes (7)
    action_dim = len(env.speed_changes) * len(env.heading_changes)
    
    agent = MaritimeDoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0005,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        target_update=1_000,
        weights=None  # Start from scratch or specify path
    )
    
    print()
    print(f"Environment setup:")
    print(f"  Map size: {map_size}")
    print(f"  Max steps: {max_steps}")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Number of ships: {num_ships}")
    print(f"  K nearest ships: {k_nearest}")
    print(f"  Speed changes: {env.speed_changes}")
    print(f"  Heading changes: {env.heading_changes}")
    print()
    
    # Training parameters
    num_episodes = 10_000
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = 500_000
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay
    
    # Training statistics
    episode_rewards = []
    collision_counts = []
    
    print("Starting training...")
    
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        # Reset environment - generates new random ship positions
        env.env.reset()
        
        state = env.get_state_ship(controlled_ship_idx)
        
        if state is None:
            print(f"Warning: Could not get initial state for episode {episode}")
            continue
        
        total_reward = 0
        step_count = 0
        done = False
        episode_actions = []
        episode_rewards_list = []
        termination_reason = None

        while not done and step_count < max_steps:
            # Select and execute action
            action = agent.select_action(state, epsilon)
            episode_actions.append(action)
            
            # Execute step
            next_state, reward, done, termination_reason = env.step(action)
            episode_rewards_list.append(reward)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update model if enough samples
            if len(agent.memory) >= agent.batch_size:
                agent.update_model()
            
            # Update statistics
            total_reward += reward
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
            
            state = next_state
            step_count += 1
        
        # Record statistics
        episode_rewards.append(total_reward)
        collision_counts.append(1 if termination_reason == "Collision" else 0)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward  = np.mean(episode_rewards_list)            
            buffer_size = len(agent.memory)
            
            print(f"Episode {episode}/{num_episodes} | Steps: {step_count} | Reason: {termination_reason}")
            print(f"  Avg reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
            print(f"  Buffer: {buffer_size}/{agent.memory.maxlen}")
        
            # save model
            name = f"models/v1_only_target_model{episode}.pth"
            agent.save_model(name)
    
    print("Training completed!")