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


class TestAgent(unittest.TestCase):

    def test_episodeCount(self):
        strategy: Strategy = QLearning(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        agent: Agent = Agent(strategy)
        self.assertEqual(agent.episodes.__len__(), 1000)

    def test_strategy(self):
        strategy: Strategy = QLearning(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        agent: Agent = Agent(strategy)
        self.assertEqual(agent.strategy.__class__.__name__, 'QLearning')
