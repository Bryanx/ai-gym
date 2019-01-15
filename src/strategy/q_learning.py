from gym import Env

from src.model.percept import Percept
from src.strategy.n_step_q_learning import NStepQLearning
from src.strategy.strategy import Strategy


class QLearning(NStepQLearning):

    def __init__(self, env: Env, episode_count: int):
        super().__init__(env, episode_count, 1)

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self.update_q_value(percept)
        self.update_v_values()
