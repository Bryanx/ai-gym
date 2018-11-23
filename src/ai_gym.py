from src.model.agent import Agent

if __name__ == '__main__':
    agent = Agent('FrozenLake-v0', episode_count=1000)
    agent.learn()
