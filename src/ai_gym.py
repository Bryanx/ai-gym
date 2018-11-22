from src.agent import Agent

if __name__ == '__main__':
    agent = Agent('FrozenLake-v0', episode_count=500)
    agent.learn()
