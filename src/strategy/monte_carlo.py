from gym import Env

from src.model.percept import Percept
from src.strategy.n_step_q_learning import NstepQlearning


class MonteCarlo(NstepQlearning):
    def __init__(self, env: Env, episode_count: int):
        super().__init__(env, episode_count)
        self.n = self.episode_count
        self.p = []  # list of percepts, used as a buffer

    def evaluate(self, percept: Percept):
        pass

    def improve(self):
        pass
