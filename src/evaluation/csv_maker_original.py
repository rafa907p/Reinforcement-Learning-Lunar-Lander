import gymnasium as gym
from stable_baselines3 import TD3
import os
import sys
import numpy as np
import csv

# Ensure that src is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

def evaluate_model(env, model, num_episodes=50, record_interval=5, max_steps=1000):
    """Run the model and record rewards every 5 timesteps up to 1000 steps."""
    all_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step_rewards = []
        cumulative_reward = 0.0

        for t in range(1, max_steps + 1):  # Loop for 1000 timesteps
            if not done and not truncated:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += reward  # Keep accumulating rewards

            if t % record_interval == 0:  # Record every 5 steps
                step_rewards.append(cumulative_reward)  # Append cumulative reward so far

        # Pad zeros if the episode ends early
        while len(step_rewards) < 200:
            step_rewards.append(cumulative_reward)  # Maintain final reward

        all_rewards.append(step_rewards)
        print(f"Episode {episode + 1}: Total Reward = {cumulative_reward:.2f}")

    return all_rewards

def compute_best_worst_mean(all_rewards):
    """Find the best, worst, and mean rewards across all episodes."""
    total_rewards = [sum(rewards) for rewards in all_rewards]
    best_idx = np.argmax(total_rewards)
    worst_idx = np.argmin(total_rewards)

    best_rewards = all_rewards[best_idx]
    worst_rewards = all_rewards[worst_idx]
    mean_rewards = np.mean(all_rewards, axis=0)

    return best_rewards, worst_rewards, mean_rewards

def save_rewards_to_csv(best, worst, mean, output_path):
    """Save the best, worst, and mean timestep rewards to a CSV file."""
    header = [f"Timestep_{i * 5}" for i in range(1, 201)]  # 200 intervals (5 to 1000)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Category"] + header)  # Write the header
        writer.writerow(["Best"] + best)
        writer.writerow(["Worst"] + worst)
        writer.writerow(["Mean"] + mean.tolist())

    print(f"Rewards saved to {output_path}")

def main():
    # Path to your trained TD3 model
    model_path = os.path.join(project_root, "src", "models", "td3", "original", "td3_LunarLanderContinuous-v3_trial_1.zip")

    # Load the trained TD3 model
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please check the path.")
        sys.exit(1)

    model = TD3.load(model_path)

    # Create the original LunarLander environment
    env = gym.make(
        "LunarLanderContinuous-v3",
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0
    )

    # Run evaluation
    num_episodes = 500
    print(f"Running {num_episodes} evaluation episodes...")
    all_rewards = evaluate_model(env, model, num_episodes)

    # Compute best, worst, and mean rewards
    best_rewards, worst_rewards, mean_rewards = compute_best_worst_mean(all_rewards)

    # Save the rewards to CSV
    output_dir = os.path.join(project_root, "data", "reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "best_worst_mean_rewards_original.csv")
    save_rewards_to_csv(best_rewards, worst_rewards, mean_rewards, output_path)

    env.close()

if __name__ == "__main__":
    main()
