import gymnasium as gym
from stable_baselines3 import TD3
import os
import sys
import numpy as np
import json
import imageio

# Ensure that src is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)


def register_environment():
    """Register the custom LunarLander environment."""
    gym.register(
        id='CustomLunarLander-v3',
        entry_point='src.environments.custom_lunar_lander:CustomLunarLander',
        max_episode_steps=1000,
        reward_threshold=200,
    )


def evaluate_model(env, model, num_episodes=50):
    """Run the model for a specified number of episodes and record rewards and frames."""
    total_rewards = []
    episode_frames = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        step = 0
        frames = []

        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            # Capture the frame for rendering
            frame = env.render()
            frames.append(frame)

        total_rewards.append(episode_reward)
        episode_frames.append(frames)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Steps = {step}")

    return total_rewards, episode_frames


def save_report(rewards, output_path):
    """Save the evaluation results as JSON."""
    report = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "rewards": rewards
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to {output_path}")


def save_gif(frames, output_path):
    """Save frames as a GIF."""
    imageio.mimsave(output_path, frames, fps=30)
    print(f"GIF saved to {output_path}")


def main():
    # Register the custom environment
    register_environment()

    # Path to your trained TD3 model
    model_path = os.path.join(project_root, "src", "models", "td3", "custom", "td3_CustomLunarLander-v3_trial_3.zip")

    # Load the trained TD3 model
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please check the path.")
        sys.exit(1)

    model = TD3.load(model_path)

    # Create the environment
    env = gym.make(
        "CustomLunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.02,
        partial_observation=True,
        render_mode="rgb_array"
    )

    # Run evaluation
    num_episodes = 50
    print(f"Running {num_episodes} evaluation episodes...")
    rewards, frames = evaluate_model(env, model, num_episodes)

    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation Report:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Standard Deviation of Reward: {std_reward:.2f}")

    # Save the report
    output_dir = os.path.join(project_root, "data", "reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_report.json")
    save_report(rewards, output_path)

    # Identify the best and worst episodes
    best_idx = np.argmax(rewards)
    worst_idx = np.argmin(rewards)

    print(f"Best Episode: {best_idx + 1} with Reward = {rewards[best_idx]:.2f}")
    print(f"Worst Episode: {worst_idx + 1} with Reward = {rewards[worst_idx]:.2f}")

    # Save GIFs of the best and worst episodes
    video_dir = os.path.join(project_root, "data", "videos")
    os.makedirs(video_dir, exist_ok=True)

    save_gif(frames[best_idx], os.path.join(video_dir, "best_episode.gif"))
    save_gif(frames[worst_idx], os.path.join(video_dir, "worst_episode.gif"))

    env.close()


if __name__ == "__main__":
    main()