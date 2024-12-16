# train_ddpg.py
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import sys

# Ensure that src is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
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
    MODEL_DIR = "../models/ddpg/"
    LOG_DIR = "../data/logs/ddpg/"
    NUM_EPISODES = 10
    MAX_TIMESTEPS = 1_000_000
    STEP_INCREMENT = 20_000
    EVAL_FREQUENCY = 10
    GOAL_REWARD = 200.0

    print("Starting DDPG training on Custom Lunar Lander environment (continuous mode).")

    # Register the custom environment
    register_environment()

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Directories '{MODEL_DIR}' and '{LOG_DIR}' are ready.")

    # Create the training environment without Monitor
    train_env = gym.make(
        "CustomLunarLander-v3",
        continuous=True,  # DDPG needs continuous actions
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.02,
        partial_observation=True
    )
    print("Training environment created for DDPG.")

    # Initialize the DDPG agent
    model = DDPG(
        'MlpPolicy',
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=(1, 'step'),
        gradient_steps=1,
        tensorboard_log=LOG_DIR,
        seed=42
    )
    print("DDPG agent initialized.")

    # Create the evaluation environment without Monitor
    eval_env = gym.make(
        "CustomLunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.02,
        partial_observation=True
    )
    print("Evaluation environment created.")

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=GOAL_REWARD, verbose=1),
        eval_freq=STEP_INCREMENT * EVAL_FREQUENCY,
        n_eval_episodes=NUM_EPISODES,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        deterministic=True,
        render=False
    )
    print("Evaluation callback defined for DDPG.")

    # Train the model
    try:
        print(f"Starting DDPG training for a total of {MAX_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=MAX_TIMESTEPS, callback=eval_callback)
        print("DDPG training completed successfully.")
    except KeyboardInterrupt:
        print("DDPG training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during DDPG training: {e}")
    finally:
        # Save the final model
        final_model_path = os.path.join(MODEL_DIR, "ddpg_custom_lunar_lander")
        model.save(final_model_path)
        print(f"Final DDPG model saved to {final_model_path}.zip")

        # Close environments
        train_env.close()
        eval_env.close()
        print("Environments closed for DDPG.")

if __name__ == "__main__":
    main()
