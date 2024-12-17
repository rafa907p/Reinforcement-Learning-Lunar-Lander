# train_td3.py
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Ensure that src is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Windows Version
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
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
    MODEL_DIR = "../models/td3/"
    LOG_DIR = "../data/logs/td3/"
    NUM_EPISODES = 10
    MAX_TIMESTEPS = 1_000_000
    STEP_INCREMENT = 20_000
    EVAL_FREQUENCY = 10
    GOAL_REWARD = 200.0

    print("Starting TD3 training on Custom Lunar Lander environment (continuous mode).")

    # Register the custom environment
    register_environment()

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Directories '{MODEL_DIR}' and '{LOG_DIR}' are ready.")

    # Create the training environment and wrap it with Monitor
    train_env = Monitor(
        gym.make(
            "CustomLunarLander-v3",
            continuous=True,  # Set to continuous for TD3
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        ),
        filename=os.path.join(LOG_DIR, "monitor_train.csv")
    )
    print("Training environment created and wrapped with Monitor for TD3.")

    # Initialize the TD3 agent
    model = TD3(
        'MlpPolicy',
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=(1, 'step'),
        gradient_steps=1,
        tensorboard_log=LOG_DIR,
        seed=42
    )
    print("TD3 agent initialized.")

    # Create a separate evaluation environment and wrap with Monitor
    eval_env = Monitor(
        gym.make(
            "CustomLunarLander-v3",
            continuous=True,  # Ensure continuous for evaluation
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        ),
        filename=os.path.join(LOG_DIR, "monitor_eval.csv")
    )
    print("Evaluation environment created and wrapped with Monitor.")

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
    print("Evaluation callback defined for TD3.")

    # Train the model
    try:
        print(f"Starting TD3 training for a total of {MAX_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=MAX_TIMESTEPS, callback=eval_callback)
        print("TD3 training completed successfully.")
    except KeyboardInterrupt:
        print("TD3 training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during TD3 training: {e}")
    finally:
        final_model_path = os.path.join(MODEL_DIR, "td3_custom_lunar_lander")
        model.save(final_model_path)
        print(f"Final TD3 model saved to {final_model_path}.zip")

        train_env.close()
        eval_env.close()
        print("Environments closed for TD3.")

if __name__ == "__main__":
    main()
