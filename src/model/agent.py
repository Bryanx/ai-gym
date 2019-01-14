import matplotlib.pyplot as plt
import numpy as np

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class Agent:
    def __init__(self, strategy: Strategy):
        self.env = strategy.env
        self.episodes = range(strategy.episode_count)
        self.strategy: Strategy = strategy

    def learn(self):
        for n in self.episodes:
            old_state = self.env.reset()
            print("Episode: " + str(n))
            for t in range(100):
                action = self.strategy.next_action(old_state)  # strategy should decide next action here
                new_state, reward, done, info = self.env.step(action)
                percept = Percept(old_state, reward, action, new_state, done, info.get("prob"))
                self.env.render()
                self.strategy.learn(percept)
                old_state = new_state
                if done:
                    print(f"Episode {n} finished after {t+1} timesteps")
                    break

        # printing policy and showing heatmap:
        self.print_policy()
        self.plot_heatmap()

    def plot_heatmap(self):
        ncol = self.env.unwrapped.ncol
        nrow = self.env.unwrapped.nrow
        x = self.strategy.v.reshape(nrow, ncol)
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='hot', interpolation='nearest')
        for i in range(0, nrow):
            for j in range(0, ncol):
                ax.text(j, i, round(x[i, j], 4), ha="center", va="center", color="lightgrey")
        plt.title(self.strategy.__class__.__name__)
        plt.show()

    def print_policy(self):
        i = 0
        directions = np.chararray(self.strategy.mdp.S.__len__(), unicode=True)
        for policy_row in self.strategy.π:
            a = np.argmax(policy_row)
            if a == 0:
                directions[i] = '🡐'
            elif a == 1:
                directions[i] = '🡓'
            elif a == 2:
                directions[i] = '🡒'
            else:
                directions[i] = '🡑'
            i += 1
        ncol = self.env.unwrapped.ncol
        nrow = self.env.unwrapped.nrow
        directions = directions.reshape(nrow, ncol)
        print(directions)
        print(self.strategy.π)
