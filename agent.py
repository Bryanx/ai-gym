import numpy as np
import matplotlib.pyplot as plt
import gym

from percept import Percept
from strategies.valueIterationStrategy import Strategy

class Agent:
    def __init__(self, gym_name: str, episodes: int):
        self.env = gym.make(gym_name)
        self.episodes = range(episodes)
        self.strategy = Strategy()

    def learn(self):
        for n in self.episodes:
            old_state = self.env.reset()
            print("Episode: " + str(n))
            for t in range(100):
                action = self.strategy.next_action()  # strategy should decide next action here
                new_state, reward, done, info = self.env.step(action)
                percept = Percept(old_state, reward, action, new_state, done, info.get("prob"))
                print(percept)
                self.env.render()
                self.strategy.learn(percept)
                old_state = new_state
                if done:
                    print(f"Episode {n} finished after {t+1} timesteps")
                    break
        self.print()


    def print(self):
        # testing / printing:
        v = self.strategy.v
        x = np.array([v[0:4], v[4:8], v[8:12], v[12:16]])
        test = self.strategy.mdp
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='hot', interpolation='nearest')
        for i in range(0, 4):
            for j in range(0, 4):
                ax.text(j, i, round(x[i, j], 4),
                        ha="center", va="center", color="lightgrey")
        plt.show()
