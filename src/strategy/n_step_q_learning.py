from src.model.percept import Percept
from src.strategy.strategy import Strategy


class NstepQlearning(Strategy):

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        # buffer all percepts:
        self.p = [percept] + self.p
        print(self.v)
        # clear buffer:
        if len(self.p) > self.n:
            for p_item in self.p:
                s = p_item.oldState
                a = p_item.action
                r = p_item.reward
                q = self.q
                max_q_next_state = q[percept.nextState, 0:self.action_count].max()
                q[s, a] -= self.α * (q[s, a] - (r + self.γ * max_q_next_state))

    def improve(self):
        # update v-values to match max-q values
        for i in range(0, self.state_count):
            self.v[i] = self.q[i, 0:self.action_count].max()
