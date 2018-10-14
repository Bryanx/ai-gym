import gym
import random
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, gymName: str, episodes: int):
        self.env = gym.make(gymName)
        self.episodes = range(episodes)
        self.strategy = Strategy()

    def learn(self):
        for n in self.episodes:
            oldState = self.env.reset()
            print("Episode: " + str(n))
            for t in range(100):
                action = self.strategy.next_action()  # strategy should decide next action here
                newState, reward, done, info = self.env.step(action)
                percept = Percept(oldState, reward, action, newState, done, info.get("prob"))
                print(percept)
                self.env.render()
                self.strategy.learn(percept)
                oldState = newState
                if done:
                    print(f"Episode {n} finished after {t+1} timesteps")
                    break
        self.test()

    def test(self):
        # testing / printing:
        v = self.strategy.v
        x = np.array([v[0:4], v[4:8], v[8:12], v[12:16]])
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='hot', interpolation='nearest')
        for i in range(0, 4):
            for j in range(0, 4):
                ax.text(j, i, round(x[i, j], 4),
                        ha="center", va="center", color="b")
        plt.show()


class MarkovDecisionProcess:
    def __init__(self, amountOfStates: int, amountOfActions: int):
        states = np.arange(0, amountOfStates, 1)
        self.rewardPerState = np.transpose((states, np.zeros(amountOfStates)))
        self.actions = np.arange(0, amountOfActions, 1)
        self.stateTransitionModel = np.zeros((amountOfStates, amountOfActions, amountOfStates))

    def update(self, percept):
        self.stateTransitionModel[percept.oldState][percept.action][percept.nextState] = round(percept.prob, 2)

    def __str__(self) -> str:
        return "Actions: \n" + str(self.stateTransitionModel) + "\nRewardPerState:\n" + str(self.rewardPerState)


class Percept:
    def __init__(self, oldState: int, reward: int, action: int, nextState: int, finished: bool, prob: float):
        self.oldState = oldState
        self.reward = reward
        self.action = action
        self.nextState = nextState
        self.finished = finished
        self.prob = prob

    def __str__(self):
        actionStr = ""
        if self.action == 0:
            actionStr = "left"
        elif self.action == 1:
            actionStr = "down"
        else:
            actionStr = "right"
        return f'From {self.oldState} pressed {actionStr}, went to {self.nextState}, got reward {self.reward}'


class Strategy:

    def __init__(self):
        self.mdp = MarkovDecisionProcess(16, 4)
        self.v = np.zeros(16)
        self.q = np.zeros((16, 4))
        self.gamma = 0.9  # discount factor -> determines the importance of future rewards.
        self.alpha = 0.1  # learning rate, determines to what extent new info override old info.

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.actions)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        s = percept.oldState
        sn = percept.nextState
        a = percept.action
        r = percept.reward
        self.q[s, a] += self.alpha * (r + self.gamma * (self.q[sn, 0:4].max() - self.q[s, a]))

        for i in range(0, 16):
            self.v[i] = self.q[i, 0:4].max()

    def improve(self):
        # improve policy here
        pass


agent = Agent('FrozenLake-v0', episodes=1000)
agent.learn()
