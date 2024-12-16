# tune_hyperparams.py
import optuna
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys

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


def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_uniform('ent_coef', 0.0, 0.01)

    # You can add more or vary policies, etc.

    env = gym.make(
        "CustomLunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.02,
        partial_observation=True
    )

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        verbose=0,
        seed=42
    )

    # Train for a smaller number of timesteps to speed up tuning
    # Adjust according to available resources and desired speed
    model.learn(total_timesteps=100_000)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

    env.close()

    # We want to maximize mean_reward, so return negative for minimization problem
    return mean_reward


def main():
    register_environment()

    # Create an Optuna study
    # You can specify a storage (e.g., sqlite) if you want to persist results
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Study completed!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
