# evaluate.py

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import os
import sys
import numpy as np
import pandas as pd

# Fix the path issue
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

# Import the custom environment
try:
    from src.environments.custom_lunar_lander import CustomLunarLander
except ImportError as e:
    print("Error importing CustomLunarLander:", e)
    sys.exit(1)


def register_custom_environment():
    """
    Register the custom Lunar Lander environment if not already registered.
    """
    try:
        gym.make('CustomLunarLander-v3')
    except gym.error.UnregisteredEnv:
        register(
            id='CustomLunarLander-v3',
            entry_point='environments.custom_lunar_lander:CustomLunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )
        print("Registered 'CustomLunarLander-v3' environment.")
    else:
        print("'CustomLunarLander-v3' environment is already registered.")


def create_environment(env_id, continuous=True, partial_observation=False, render_mode=None):
    """
    Create and return a Gymnasium environment.
    """
    env = gym.make(
        env_id,
        continuous=continuous,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.02,
        partial_observation=partial_observation,
        render_mode=render_mode
    )
    return env


def evaluate_agent(env, model, num_episodes=100):
    """
    Evaluate the trained agent on the given environment.
    """
    rewards, steps = [], []
    success_count = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward, step_count = 0.0, 0
        done, truncated = False, False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)
        if total_reward >= env.spec.reward_threshold:
            success_count += 1

    metrics = {
        'average_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'average_steps': np.mean(steps),
        'success_rate': (success_count / num_episodes) * 100
    }
    return metrics


def main():
    # Register the custom environment
    register_custom_environment()

    # Paths to trained models and logs
    models_dir = "../models/ddpg/"
    logs_dir = "../data/logs/"
    os.makedirs(logs_dir, exist_ok=True)

    original_env_id = 'LunarLanderContinuous-v2'  # Original Gym environment (continuous)
    custom_env_id = 'CustomLunarLander-v3'        # Custom environment

    # Load pre-trained models
    original_model_path = os.path.join(models_dir, "best_model.zip")
    custom_model_path = os.path.join(models_dir, "ppo_custom_lander.zip")

    print("\nLoading pre-trained models...")
    try:
        original_model = PPO.load(original_model_path)
        print("Loaded PPO model for Original Environment.")
        custom_model = PPO.load(custom_model_path)
        print("Loaded PPO model for Custom Environment.")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # Create environments
    print("Creating Original Lunar Lander Environment...")
    original_env = create_environment(original_env_id, continuous=True)
    print("Creating Custom Lunar Lander Environment...")
    custom_env = create_environment(custom_env_id, continuous=True, partial_observation=True)

    # Evaluate models
    print("\nEvaluating PPO on Original Environment...")
    original_metrics = evaluate_agent(original_env, original_model)
    print("Original Environment Metrics:", original_metrics)

    print("\nEvaluating PPO on Custom Environment...")
    custom_metrics = evaluate_agent(custom_env, custom_model)
    print("Custom Environment Metrics:", custom_metrics)

    # Save comparison results
    metrics_df = pd.DataFrame({
        'Environment': ['Original', 'Custom'],
        'Average Reward': [original_metrics['average_reward'], custom_metrics['average_reward']],
        'Reward Std Dev': [original_metrics['std_reward'], custom_metrics['std_reward']],
        'Success Rate (%)': [original_metrics['success_rate'], custom_metrics['success_rate']]
    })

    metrics_path = os.path.join(logs_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nEvaluation metrics saved to {metrics_path}")

    # Cleanup
    original_env.close()
    custom_env.close()
    print("\nEnvironments closed. Evaluation complete.")


if __name__ == "__main__":
    main()
