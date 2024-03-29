import numpy as np
from gym import Env
from math import inf

from src.model.percept import Percept
from src.strategy.strategy import Strategy


class ValueIteration(Strategy):

    def __init__(self, env: Env, episode_count: int):
        super().__init__(env, episode_count)
        self.ξ = 0.1  # The v-values precession we need to achieve
        self.Δ = None  # Delta, absolute difference between 2 v-values from one state

    # Value iteration
    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        rmax = np.amax(self.mdp.rewardPerState)
        self.Δ = inf
        while self.Δ > self.ξ * rmax * (1 - self.γ / self.γ):
            self.Δ = 0
            for s in self.mdp.S:
                old_value = self.v[s]
                self.v[s] = self.value_function(s)
                self.Δ = max(self.Δ, abs(old_value - self.v[s]))

    def value_function(self, s: int):
        A = self.get_action_values(s)
        return round(np.max(A), 5)

    # policy improvement will use this function.
    def get_best_action_for_state(self, s: int):
        A = self.get_action_values(s)
        return np.random.choice(np.flatnonzero(A == A.max()))

    # returns the v-values for each action given a certain state.
    def get_action_values(self, s: int):
        A = np.zeros(self.action_count)
        for a in self.mdp.A:
            for next_s in self.mdp.S:
                prob = self.mdp.stateTransitionModel[s, a, next_s]
                reward = self.mdp.rewardPerState[next_s]
                A[a] += prob * (reward + self.γ * self.v[next_s])
        return A
