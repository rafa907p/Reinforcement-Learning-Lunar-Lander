# src/environments/custom_lunar_lander.py

import gymnasium as gym
from gymnasium.envs.box2d import LunarLander
import numpy as np

class CustomLunarLander(LunarLander):
    def __init__(
        self,
        continuous=False,
        gravity=-10.0,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.0,
        observation_noise=0.01,
        partial_observation=False,
        render_mode=None  # Add render_mode parameter
    ):
        super(CustomLunarLander, self).__init__(
            continuous=continuous,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            render_mode=render_mode  # Pass render_mode to superclass
        )
        self.observation_noise = observation_noise
        self.partial_observation = partial_observation

        if self.partial_observation:
            # Update observation_space by removing indices [5,6]
            original_low = self.observation_space.low
            original_high = self.observation_space.high
            # Remove angular velocity (index 5) and left_leg (index 6)
            new_low = np.delete(original_low, [5, 6])
            new_high = np.delete(original_high, [5, 6])
            self.observation_space = gym.spaces.Box(
                low=new_low,
                high=new_high,
                dtype=np.float32
            )

    def step(self, action):
        # Unpack the five return values from the parent class
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated  # Determine if the episode is done

        # Modify rewards
        if action == 2:  # main engine
            reward -= 0.5  # originally was -0.3

        # Punish the agent for hovering
        if len(obs) >= 8:  # Ensure obs has enough elements
            x, y, vx, vy, theta, vtheta, left_leg, right_leg = obs
            if y > 0.5 and abs(vx) < 0.1 and abs(vy) < 0.1:
                reward -= 0.1

        # Add observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, size=obs.shape)
            obs = obs + noise

        # Apply partial observability: Remove specific state variables
        if self.partial_observation:
            # Remove angular velocity and left_leg (indices 5 and 6)
            if len(obs) > 6:
                obs = np.delete(obs, [5, 6])

        # Ensure dtype float32
        obs = obs.astype(np.float32)

        # Clip observations to stay within observation_space
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Unpack the two return values from the parent class
        obs, info = super().reset(**kwargs)

        # Apply observation noise on reset
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, size=obs.shape)
            obs = obs + noise

        # Apply partial observability
        if self.partial_observation:
            # Remove angular velocity and left_leg (indices 5 and 6)
            if len(obs) > 6:
                obs = np.delete(obs, [5, 6])

        # Ensure dtype float32
        obs = obs.astype(np.float32)

        # Clip observations to stay within observation_space
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs, info
