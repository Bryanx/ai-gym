from src.model.agent import Agent
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196
)

if __name__ == '__main__':
    agent = Agent('FrozenLakeNotSlippery-v0', episode_count=1000)
    agent.learn()
