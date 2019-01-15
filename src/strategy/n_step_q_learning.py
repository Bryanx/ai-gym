from gym import Env

from src.model.percept import Percept
from src.strategy.q_learning import QLearning
from src.strategy.strategy import Strategy


class NStepQLearning(QLearning):

    def __init__(self, env: Env, episode_count: int, number_of_n: int = 3):
        super().__init__(env, episode_count)
        self.n = number_of_n  # amount of steps
        self.P = []  # list of percepts, used as a buffer

    def evaluate(self, percept: Percept):
        # Update mdp wih percept
        self.mdp.update(percept)

        # buffer all percepts:
        self.P.insert(0, percept)

        # clear buffer:
        if len(self.P) >= self.n:
            self.update_q_values()
            self.P.pop()

    def update_q_values(self):
        for p in self.P:
            self.update_q_value(p)
        self.update_v_values()
