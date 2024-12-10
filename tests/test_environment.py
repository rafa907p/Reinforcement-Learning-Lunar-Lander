# tests/test_environment.py

import unittest
import gymnasium as gym
from gymnasium.envs.registration import register
import os
import sys

# Add the parent directory to the Python path to allow imports from 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)



class TestCustomLunarLander(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Register the custom environment once before all tests.
        """
        register(
            id='CustomLunarLander-v3',
            entry_point='src.environments.custom_lunar_lander:CustomLunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )

    def setUp(self):
        """
        Create a fresh environment instance before each test.
        """
        self.env = gym.make(
            "CustomLunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=1.0,
            observation_noise=0.02,
            partial_observation=True
        )

    def test_reset(self):
        """
        Test if the environment resets correctly.
        """
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_step(self):
        """
        Test if the environment steps correctly.
        """
        obs, info = self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def tearDown(self):
        """
        Close the environment after each test.
        """
        self.env.close()


if __name__ == '__main__':
    unittest.main()
