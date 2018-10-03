import gym
import random
import numpy as np


class Agent:
    def __init__(self, gymName: str, episodes: int):
        self.env = gym.make(gymName)
        self.episodes = range(episodes)
        self.mdp = MarkovDecisionProcess(16, 4)

    def learn(self):
        for n in self.episodes:
            oldState = self.env.reset()
            print("Episode: " + str(n))
            for t in range(100):
                action = random.choice(self.mdp.actions)  # strategy should decide next action here
                newState, reward, done, info = self.env.step(action)
                percept = Percept(oldState, reward, action, newState, done, info.get("prob"))
                print(percept)
                self.env.render()
                self.mdp.update(percept)
                oldState = newState
                if done:
                    print(f"Episode {n} finished after {t+1} timesteps")
                    break

        # test here
        print(self.mdp.stateTransitionModel[0][1][1])


class MarkovDecisionProcess:
    def __init__(self, amountOfStates: int, amountOfActions: int):
        states = np.arange(0, amountOfStates, 1)
        self.rewardPerState = np.transpose((states, np.zeros(amountOfStates)))
        self.actions = np.arange(0, amountOfActions, 1)
        self.stateTransitionModel = np.zeros((amountOfStates, amountOfActions, amountOfStates))

    def update(self, percept):
        self.stateTransitionModel[percept.oldState][percept.action][percept.nextState] = round(percept.prob,2)

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


agent = Agent('FrozenLake-v0', episodes=100)
agent.learn()
