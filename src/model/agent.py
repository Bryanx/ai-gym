import matplotlib.pyplot as plt
import gym

from src.model.percept import Percept
from src.strategy.q_learning import Qlearning


class Agent:
    def __init__(self, gym_name: str, episode_count: int):
        self.env = gym.make(gym_name)
        self.episodes = range(episode_count)
        self.strategy = Qlearning(self.env, episode_count)

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

    # testing / printing:
    def print(self):
        ncol = self.env.unwrapped.ncol
        nrow = self.env.unwrapped.nrow
        x = self.strategy.v.reshape(nrow, ncol)
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='hot', interpolation='nearest')
        for i in range(0, nrow):
            for j in range(0, ncol):
                ax.text(j, i, round(x[i, j], 4), ha="center", va="center", color="lightgrey")
        plt.show()
