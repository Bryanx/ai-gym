from agent import Agent

if __name__ == '__main__':
    agent = Agent('FrozenLake-v0', episodes=1000)
    agent.learn()
