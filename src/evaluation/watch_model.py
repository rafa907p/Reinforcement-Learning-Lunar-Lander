# src/evaluation/watch_model.py

import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import imageio

# Ensure that src is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

from src.environments.custom_lunar_lander import CustomLunarLander

def register_environment():
    gym.register(
        id='CustomLunarLander-v3',
        entry_point='src.environments.custom_lunar_lander:CustomLunarLander',
        max_episode_steps=1000,
        reward_threshold=200,
    )

def main():
    # Register the custom environment
    register_environment()

    # Path to your saved model
    model_path = "../../models/ppo_custom_lunar_lander.zip"  # Adjust as needed

    # Load the trained PPO model
    model = PPO.load(model_path)

    # Create the environment with render_mode set to 'rgb_array'
    env = gym.make(
        "CustomLunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.02,
        partial_observation=True,
        render_mode="rgb_array"  # Change to 'rgb_array' for frame capture
    )

    num_episodes = 5  # Number of episodes to watch

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        frames = []

        while not done and not truncated:
            # Use the trained model to predict the action
            action, _states = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Capture the frame
            frame = env.render()
            frames.append(frame)

        # Save the episode as a GIF
        video_dir = "../../data/videos/"
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"episode_{episode + 1}.gif")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}, Video saved to {video_path}")

    env.close()

if __name__ == "__main__":
    main()
