from src.percept import Percept
from src.strategies.strategy import Strategy


class Qlearning(Strategy):

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        s = percept.oldState
        a = percept.action
        r = percept.reward
        q = self.q
        max_q_next_state = q[percept.nextState, 0:self.action_count].max()
        # q of current state = reward + max-q of next state    (gamma and alpha tweak the equation)
        q[s, a] += self.α * (r + self.γ * (max_q_next_state - q[s, a]))

    def improve(self):
        # update v-values to match max-q values
        for i in range(0, self.state_count):
            self.v[i] = self.q[i, 0:self.action_count].max()

