from gym import Env

from src.model.percept import Percept
from src.strategy.n_step_q_learning import NstepQlearning


class MonteCarlo(NstepQlearning):
    def __init__(self, env: Env, episode_count: int):
        super().__init__(env, episode_count, episode_count)

    def evaluate(self, percept: Percept):
        # Update mdp wih percept
        self.mdp.update(percept)

        # buffer all percepts:
        self.P = [percept] + self.P

        # clear buffer if agent has finished episode:
        if percept.finished:
            self.update_q_values()
            self.P = []
