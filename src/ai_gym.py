import gym
from src.model.agent import Agent
from gym.envs.registration import register
from src.strategy.strategy import Strategy
from src.strategy.q_learning import QLearning
from src.strategy.n_step_q_learning import NStepQLearning
from src.strategy.value_iteration import ValueIteration
from src.strategy.monte_carlo import MonteCarlo

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196,  # optimum = .8196
)

if __name__ == '__main__':
    strategy: Strategy = NStepQLearning(gym.make('FrozenLakeNotSlippery-v0'), episode_count=1000)
    agent = Agent(strategy)
    agent.learn()
