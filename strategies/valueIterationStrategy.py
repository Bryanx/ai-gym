import random
import numpy as np

from models.mdp import MarkovDecisionProcess
from percept import Percept


class Strategy:
    def __init__(self):
        self.mdp = MarkovDecisionProcess(16, 4)
        self.v = np.zeros(16)  # v values -> this will decide the policy - the max q value of each state.
        self.q = np.zeros((16, 4))  # -> q values -> for each state-action the average reward to the goal.
        self.γ = 0.9  # discount factor -> determines the importance of future rewards.
        self.α = 0.1  # learning rate, determines to what extent new info overrides old info.
        self.π = np.full((16, 4), 0.25)
        self.ξ = 0.1  # fineness factor

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.actions)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    # Value iteration
    def evaluate(self, percept: Percept):
        self.mdp.update(percept)

        Δ = 5000000
        rmax = np.amax(self.mdp.rewardPerState)
        while Δ > self.ξ * rmax * (1 - self.γ / self.γ):
            Δ = 0
            for s in range(0, 16):
                old_value = self.v[s]
                self.v[s] = self.valueFunction(s)
                Δ = abs(old_value - self.v[s])

    def valueFunction(self, s: int):
        newV = self.v[s]
        A = np.zeros(4)
        for a in range(0, 4):
            for next_s in range(0, 16):
                prob = self.mdp.stateTransitionModel[s, a, next_s]
                reward = self.mdp.rewardPerState[next_s]
                A[a] += prob * (reward + self.γ * self.v[next_s])
            newV = round(np.max(A),5)
        return newV

    def improve(self):
        pass
