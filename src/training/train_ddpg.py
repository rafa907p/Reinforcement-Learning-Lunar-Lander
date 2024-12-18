import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import optuna
import os
import sys
import json

# Fix the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the custom environment
try:
    from src.environments.custom_lunar_lander import CustomLunarLander
except ImportError as e:
    print("Error importing CustomLunarLander:", e)
    sys.exit(1)

# Paths
MODELS_DIR = "../models/ddpg/"
LOGS_DIR = "../data/logs/ddpg/"
ORIGINAL_ENV_ID = "LunarLanderContinuous-v3"
CUSTOM_ENV_ID = "CustomLunarLander-v3"
GOAL_REWARD = 200.0


# Register the custom environment
def register_custom_environment():
    try:
        gym.make(CUSTOM_ENV_ID)
    except gym.error.UnregisteredEnv:
        register(
            id=CUSTOM_ENV_ID,
            entry_point="src.environments.custom_lunar_lander:CustomLunarLander",
            max_episode_steps=1000,
            reward_threshold=GOAL_REWARD,
        )
        print(f"Registered '{CUSTOM_ENV_ID}' environment.")
    else:
        print(f"'{CUSTOM_ENV_ID}' environment is already registered.")


# Create environment with selective arguments
def create_environment(env_id, **kwargs):
    if env_id == CUSTOM_ENV_ID:
        # Pass custom arguments only to the Custom Lunar Lander
        return gym.make(env_id, **kwargs)
    else:
        # For the original environment, do not pass unsupported arguments
        return gym.make(env_id)


# Save metrics to a JSON file
def save_metrics(metrics, file_path):
    """
    Save evaluation metrics to a JSON file.

    Parameters:
        metrics (dict): Dictionary containing metrics.
        file_path (str): Path to save the metrics file.
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")


# Define Optuna objective function for hyperparameter tuning
def objective(trial, env_id, model_dir, log_dir):
    env = create_environment(env_id, continuous=True, partial_observation=True, observation_noise=0.02)

    # Suggest hyperparameters using Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    tau = trial.suggest_float("tau", 0.005, 0.05)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    buffer_size = trial.suggest_categorical("buffer_size", [100000, 500000, 1000000])

    # Initialize the DDPG model
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        buffer_size=buffer_size,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=0,
        tensorboard_log=log_dir,
        seed=42,
    )

    # Callback to stop training when reward threshold is reached
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=GOAL_REWARD, verbose=1),
        eval_freq=10_000,
        n_eval_episodes=5,
        log_path=log_dir,
        deterministic=True,
        render=False,
    )

    # Train the model
    model.learn(total_timesteps=200_000, callback=eval_callback)

    # Evaluate the model
    mean_reward, total_rewards = evaluate_model(model, env)
    env.close()

    # Save the model
    model_path = os.path.join(model_dir, f"ddpg_{env_id}_trial_{trial.number}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics = {
        "trial_number": trial.number,
        "mean_reward": mean_reward,
        "total_rewards": total_rewards,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "tau": tau,
        "gamma": gamma,
        "buffer_size": buffer_size
    }
    metrics_path = os.path.join(log_dir, f"metrics_trial_{trial.number}.json")
    save_metrics(metrics, metrics_path)

    return mean_reward


def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate a trained model on a given environment.
    """
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    mean_reward = sum(total_rewards) / len(total_rewards)
    return mean_reward, total_rewards


def train_with_optuna(env_id, model_dir, log_dir, study_name):
    """
    Optimize DDPG hyperparameters using Optuna.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda trial: objective(trial, env_id, model_dir, log_dir), n_trials=10)

    # Log best trial
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save final best trial metrics
    best_metrics = {
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params
    }
    best_metrics_path = os.path.join(log_dir, f"{study_name}_best_metrics.json")
    save_metrics(best_metrics, best_metrics_path)


def main():
    # Register custom environment
    register_custom_environment()

    # Paths for models and logs
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Train on Original Lunar Lander
    print("\nTraining DDPG on Original Lunar Lander Continuous Environment...")
    train_with_optuna(
        ORIGINAL_ENV_ID,
        os.path.join(MODELS_DIR, "original"),
        os.path.join(LOGS_DIR, "original"),
        study_name="ddpg_original_lander",
    )

    # Train on Custom Lunar Lander
    print("\nTraining DDPG on Custom Lunar Lander Environment...")
    train_with_optuna(
        CUSTOM_ENV_ID,
        os.path.join(MODELS_DIR, "custom"),
        os.path.join(LOGS_DIR, "custom"),
        study_name="ddpg_custom_lander",
    )


if __name__ == "__main__":
    main()
