import unittest

import gym
from gym.envs.registration import register

from src.model.agent import Agent
from src.strategy.q_learning import QLearning
from src.strategy.strategy import Strategy

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196,  # optimum = .8196
)


class test_policy(unittest.TestCase):
    def test_policy(self):
        strategy: Strategy = QLearning(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        agent: Agent = Agent(strategy)
        agent.learn()

        for s in range(strategy.state_count):
            self.assertEqual(sum(strategy.π[s]), 1.0)
