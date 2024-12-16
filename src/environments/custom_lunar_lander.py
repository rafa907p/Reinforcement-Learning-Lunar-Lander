# src/environments/custom_lunar_lander.py

import gymnasium as gym
from gymnasium.envs.box2d import LunarLander
import numpy as np

class CustomLunarLander(LunarLander):
    def __init__(
        self,
        continuous=True,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.01,
        partial_observation=True,
        render_mode=None
    ):
        super().__init__(
            continuous=continuous,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            render_mode=render_mode
        )
        self.observation_noise = observation_noise
        self.partial_observation = partial_observation

        # If partial observation is enabled, adjust the observation space
        # Remove angular velocity (index 5) and left leg contact (index 6)
        # Original obs has shape (8,). Removing two elements -> shape (6,)
        if self.partial_observation:
            orig_low = self.observation_space.low
            orig_high = self.observation_space.high
            # Removing indices [5, 6] from original observation
            new_low = np.delete(orig_low, [5, 6])
            new_high = np.delete(orig_high, [5, 6])
            self.observation_space = gym.spaces.Box(
                low=new_low,
                high=new_high,
                dtype=np.float32
            )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Reward shaping: Penalize main engine usage
        # For continuous actions: main_thrust = action[0]
        # For discrete: main engine action = 2
        if self.continuous:
            main_thrust = action[0]
            # If the main_thrust is significantly engaged (>0.5), penalize
            if main_thrust > 0.5:
                reward -= 0.1 * main_thrust
        else:
            # In discrete mode, action==2 fires the main engine
            if action == 2:
                reward -= 0.1

        # Add observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, size=obs.shape)
            obs = obs + noise

        # Apply partial observability if enabled
        if self.partial_observation and len(obs) == 8:
            obs = np.delete(obs, [5, 6])  # remove angular velocity and left leg info

        # Clip observations
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Add observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, size=obs.shape)
            obs = obs + noise

        # Apply partial observability
        if self.partial_observation and len(obs) == 8:
            obs = np.delete(obs, [5, 6])

        # Clip observations
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

        return obs, info
