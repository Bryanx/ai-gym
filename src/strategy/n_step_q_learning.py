from gym import Env

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class NStepQLearning(Strategy):

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

    def update_q_value(self, percept: Percept):
        # fields to attributes for reading purposes
        s, a, r, q, α, γ = percept.oldState, percept.action, percept.reward, self.q, self.α, self.γ
        max_q_next_state = q[percept.nextState, 0:self.action_count].max()

        # q of current state = reward + max-q of next state (gamma and alpha tweak the equation)
        self.q[s, a] += α * (r + γ * max_q_next_state - q[s, a])

    def update_v_values(self):
        for s in self.mdp.S:
            self.v[s] = self.q[s, 0:self.action_count].max()
