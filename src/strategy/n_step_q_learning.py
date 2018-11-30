from gym import Env

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class NstepQlearning(Strategy):

    def __init__(self, env: Env, episode_count: int, number_of_n: int = 3):
        super().__init__(env, episode_count)
        self.n = number_of_n  # amount of steps
        self.P = []  # list of percepts, used as a buffer

    def evaluate(self, percept: Percept):
        # Update mdp wih percept
        self.mdp.update(percept)
        # buffer all percepts:
        self.P = [percept] + self.P
        # clear buffer:
        if len(self.P) >= self.n:
            self.update_q_values()
            self.P.pop()

    def update_q_values(self):
        for percept in self.P:
            s, a, r, q = percept.oldState, percept.action, percept.reward, self.q
            max_q_next_state = q[percept.nextState, 0:self.action_count].max()
            self.q[s, a] -= self.α * (q[s, a] - (r + self.γ * max_q_next_state))
        self.update_v_values()

    def update_v_values(self):
        for s in self.mdp.S:
            self.v[s] = self.q[s, 0:self.action_count].max()
