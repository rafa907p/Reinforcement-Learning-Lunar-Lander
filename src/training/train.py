# src/training/train_model.py

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import sys
import numpy as np

# Ensure that src is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)


def register_environment():
    if 'CustomLunarLander-v3' not in gym.envs.registry:
        register(
            id='CustomLunarLander-v3',
            entry_point='src.environments.custom_lunar_lander:CustomLunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )
    else:
        print("Environment 'CustomLunarLander-v3' is already registered.")

def main():
    # Configuration Parameters
    MODEL_DIR = "../../models/"
    LOG_DIR = "../../data/logs/"
    NUM_EPISODES = 10 # Num of times from start to land
    MAX_TIMESTEPS = 1_000_000 # Num of actions from start to land
    STEP_INCREMENT = 20_000
    EVAL_FREQUENCY = 10  # Evaluate every 10 training increments
    GOAL_REWARD = 200.0

    print("Starting training script for PPO on Custom Lunar Lander environment.")

    # Register the custom environment
    register_environment()

    # Create the training environment
    try:
        train_env = gym.make(
            "CustomLunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        )
        print("Training environment created successfully.")
    except gym.error.Error as e:
        print(f"Error creating training environment: {e}")
        sys.exit(1)

    # Initialize the PPO agent
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        tensorboard_log=os.path.join(LOG_DIR, 'ppo_lunar_lander_tensorboard'),
        seed=42
    )
    print("PPO agent initialized.")

    # Create the models and logs directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Directories '{MODEL_DIR}' and '{LOG_DIR}' are ready.")

    # Create a separate evaluation environment
    try:
        eval_env = gym.make(
            "CustomLunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        )
        print("Evaluation environment created successfully.")
    except gym.error.Error as e:
        print(f"Error creating evaluation environment: {e}")
        sys.exit(1)

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=GOAL_REWARD, verbose=1),
        eval_freq=STEP_INCREMENT * EVAL_FREQUENCY,
        n_eval_episodes=NUM_EPISODES,
        best_model_save_path=os.path.join(MODEL_DIR, 'best_model'),
        log_path=LOG_DIR,
        deterministic=True,
        render=False
    )
    print("Evaluation callback defined.")

    # Train the model with the evaluation callback
    try:
        print(f"Starting training for a total of {MAX_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=MAX_TIMESTEPS, callback=eval_callback)
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Save the final model
        final_model_path = os.path.join(MODEL_DIR, "ppo_custom_lunar_lander")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}.zip")

        # Close environments
        train_env.close()
        eval_env.close()
        print("Environments closed.")

if __name__ == "__main__":
    main()
