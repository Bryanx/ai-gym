from gym import Env

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class NstepQlearning(Strategy):
    def __init__(self, env: Env, epsiode_cound: int):
        super().__init__(env, epsiode_cound)
        self.n = 3  # amount of steps
        self.p = []  # list of percepts, used as a buffer

    def evaluate(self, percept: Percept):
        # Update mdp wih percept
        self.mdp.update(percept)

        # buffer all percepts:
        self.p = [percept] + self.p
        print(self.v)

        # clear buffer:
        if len(self.p) > self.n:
            for cur_percept in self.p:
                s = cur_percept.oldState
                a = cur_percept.action
                r = cur_percept.reward
                q = self.q

                max_q_next_state = q[percept.nextState, 0:self.action_count].max()
                q[s, a] -= self.α * (q[s, a] - (r + self.γ * max_q_next_state))

    def improve(self):
        # update v-values to match max-q values
        for i in range(0, self.state_count):
            self.v[i] = self.q[i, 0:self.action_count].max()
