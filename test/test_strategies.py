import unittest

import gym
from gym.envs.registration import register
from src.model.agent import Agent
from src.strategy.monte_carlo import MonteCarlo
from src.strategy.n_step_q_learning import NstepQlearning
from src.strategy.q_learning import QLearning
from src.strategy.strategy import Strategy
from src.strategy.value_iteration import ValueIteration

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196,  # optimum = .8196
)


class TestStrategies(unittest.TestCase):

    def test_value_iteration(self):
        strategy: Strategy = ValueIteration(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        Agent(strategy).learn()

    def test_q_learning(self):
        strategy: Strategy = QLearning(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        Agent(strategy).learn()

    def test_n_step_q_learning(self):
        strategy: Strategy = NstepQlearning(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        Agent(strategy).learn()

    def test_monte_carlo(self):
        strategy: Strategy = MonteCarlo(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
        Agent(strategy).learn()
