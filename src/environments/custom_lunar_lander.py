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

        # Adjust observation space if partial observation is enabled
        if self.partial_observation:
            orig_low = self.observation_space.low
            orig_high = self.observation_space.high
            # Remove angular velocity (index 5) and left leg contact (index 6)
            # Original obs shape: (8,) -> After removal: (6,)
            new_low = np.delete(orig_low, [5, 6])
            new_high = np.delete(orig_high, [5, 6])
            self.observation_space = gym.spaces.Box(
                low=new_low,
                high=new_high,
                dtype=np.float32
            )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Extract angle (index 4 in original observation: obs[4] = angle)
        # Note: If partial_observation is True, we must adjust the index since we removed data.
        # Original obs: x, y, vx, vy, angle, angular_vel, left_leg, right_leg
        # After removing (5, 6): we have: x, y, vx, vy, angle, right_leg
        # angle is now at index 4 in both partial and full obs since we removed 5,6, not 4.
        # Check length of obs to confirm indexing:
        # If partial_observation = True, obs length = 6 -> angle at index 4
        # If partial_observation = False, obs length = 8 -> angle at index 4 still
        angle_index = 4
        angle = obs[angle_index]

        # Reward shaping: Penalize main engine usage
        if self.continuous:
            main_thrust = action[0]
            if main_thrust > 0.5:
                reward -= 0.1 * main_thrust
        else:
            # In discrete mode, main engine is action==2
            if action == 2:
                reward -= 0.1

        # New: Angle Stability Penalty
        # Add a small penalty based on the absolute angle. The larger the angle, the bigger the penalty.
        # For example, -0.01 * abs(angle)
        reward -= 0.01 * abs(angle)

        # New: Small Time Penalty
        # Penalize each step by -0.001 to encourage faster landing.
        reward -= 0.001

        # Add observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, size=obs.shape)
            obs = obs + noise

        # Apply partial observability
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
