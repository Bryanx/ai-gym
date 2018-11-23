from gym import Env

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class NstepQlearning(Strategy):
    def __init__(self, env: Env, epsiode_cound: int, number_of_n: int = 3):
        super().__init__(env, epsiode_cound)
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
        for cur_percept in self.P:
            s = cur_percept.oldState
            a = cur_percept.action
            r = cur_percept.reward
            q = self.q

            max_q_next_state = q[cur_percept.nextState, 0:self.action_count].max()
            q[s, a] -= self.α * (q[s, a] - (r + self.γ * max_q_next_state))

    def improve(self):
        # update v-values to match max-q values
        for i in range(0, self.state_count):
            self.v[i] = self.q[i, 0:self.action_count].max()
