import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
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
    MODEL_DIR = "../models/sac_continuous/"
    LOG_DIR = "../data/logs/sac_continuous/"
    NUM_EPISODES = 10
    MAX_TIMESTEPS = 1_000_000
    STEP_INCREMENT = 20_000
    EVAL_FREQUENCY = 10
    GOAL_REWARD = 200.0

    print("Starting SAC training on Custom Lunar Lander environment (continuous mode).")

    # Register the custom environment
    register_environment()

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Directories '{MODEL_DIR}' and '{LOG_DIR}' are ready.")

    # Create the training environment and wrap with Monitor
    train_env = Monitor(
        gym.make(
            "CustomLunarLander-v3",
            continuous=True,           # Enable continuous action space
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        ),
        filename=os.path.join(LOG_DIR, "monitor_train.csv")  # monitor logs for training env
    )
    print("Training environment created and wrapped with Monitor.")

    # Initialize the SAC agent for continuous control
    model = SAC(
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
    print("SAC agent initialized for continuous control.")

    # Create the evaluation environment and wrap with Monitor
    eval_env = Monitor(
        gym.make(
            "CustomLunarLander-v3",
            continuous=True,           # Ensure continuous for evaluation as well
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        ),
        filename=os.path.join(LOG_DIR, "monitor_eval.csv")  # monitor logs for eval env
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
    print("Evaluation callback defined for SAC.")

    # Train the model
    try:
        print(f"Starting SAC training for a total of {MAX_TIMESTEPS} timesteps...")
        model.learn(total_timesteps=MAX_TIMESTEPS, callback=eval_callback)
        print("SAC training completed successfully.")
    except KeyboardInterrupt:
        print("SAC training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during SAC training: {e}")
    finally:
        # Save the final model
        final_model_path = os.path.join(MODEL_DIR, "sac_custom_lunar_lander")
        model.save(final_model_path)
        print(f"Final SAC model saved to {final_model_path}.zip")

        # Close environments
        train_env.close()
        eval_env.close()
        print("Environments closed for SAC.")

if __name__ == "__main__":
    main()