import torch

import numpy as np

def train_model():
    """Main training function"""
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Environment parameters
    k_nearest = 5
    num_ships = 3
    
    # Create environment and agent
    env = MaritimeEnvironment(
        num_ships=num_ships, 
        k_nearest=k_nearest
    )

    agent = MaritimeDoubleDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=0.0005,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        target_update=1_000,
        weights="models/ship_collision_avoidance_model150.pth"
    )
    
    print()
    print(f"Environment setup:")
    print(f"  Env size: {env.size}")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print(f"  Number of ships: {num_ships}")
    print(f"  K nearest ships: {k_nearest}")
    print()
    
    # Training parameters
    num_episodes = 10_000
    epsilon_start = 0.5
    epsilon_end = 0.05
    epsilon_decay = 100_000
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
            agent.update_model(verbose=step_verbose)
            
            # if debug_info is not None:
            #     losses.append(debug_info['loss'])
            #     grad_norms.append(debug_info['grad_norm'])
            #     q_values_log.append(debug_info['q_mean'])
            
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
            # print(f"  Loss: {recent_loss:.4f} | Grad: {recent_grad_norm:.4f} | Q-mean: {recent_q:.4f}")
            print(f"  Buffer: {buffer_size}/{agent.memory.maxlen} | Collisions: {avg_collision:.3f}")
            
            # if verbose:
            #     action_distribution = np.bincount(episode_actions, minlength=env.action_dim)
            #     most_common_actions = np.argsort(action_distribution)[-3:][::-1]
            #     print(f"  [DEBUG] Most common actions: {most_common_actions}, counts: {action_distribution[most_common_actions]}")
            #     print(f"  [DEBUG] Reward distribution: min={min(episode_rewards_list):.3f}, "
            #           f"max={max(episode_rewards_list):.3f}, mean={np.mean(episode_rewards_list):.3f}")
            # print()
            
        # Save progress
        if episode % 10 == 0:
            name = "models/v1_ship_collision_avoidance_model" + str(episode) + ".pth"
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