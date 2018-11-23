import numpy as np

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class ValueIteration(Strategy):

    # Value iteration
    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        Δ = 5000000
        rmax = np.amax(self.mdp.rewardPerState)
        while Δ > self.ξ * rmax * (1 - self.γ / self.γ):
            Δ = 0
            for s in range(0, self.state_count):
                old_value = self.v[s]
                self.v[s] = self.value_function(s)
                Δ = abs(old_value - self.v[s])

    def value_function(self, s: int):
        newV = self.v[s]
        A = np.zeros(self.action_count)
        for a in range(0, self.action_count):
            for next_s in range(0, self.state_count):
                prob = self.mdp.stateTransitionModel[s, a, next_s]
                reward = self.mdp.rewardPerState[next_s]
                A[a] += prob * (reward + self.γ * self.v[next_s])
            newV = round(np.max(A), 5)
        return newV

    def improve(self):
        pass  # TODO
