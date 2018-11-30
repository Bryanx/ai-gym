import numpy as np

from src.model.percept import Percept
from src.strategy.strategy import Strategy

class Qlearning(Strategy):

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        # fields to attributes for reading purposes
        s, a, r, q, α, γ = percept.oldState, percept.action, percept.reward, self.q, self.α, self.γ
        S = np.arange(0, self.state_count, 1)

        max_q_next_state = q[percept.nextState, 0:self.action_count].max()
        # q of current state = reward + max-q of next state    (gamma and alpha tweak the equation)
        self.q[s, a] += α * (r + γ * (max_q_next_state - q[s, a]))
        self.update_v_values()

    def update_v_values(self):
        for s in self.mdp.S:
            self.v[s] = self.q[s, 0:self.action_count].max()
