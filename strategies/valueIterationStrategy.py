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
            print(Δ)
            for state in range(0, 16):
                old_values = self.v
                self.v = self.valueFunction(percept.oldState)
                Δ = max(old_values - self.v)

    def valueFunction(self, old_state: int):
        utilValues = np.zeros(16)

        for s in range(0,16):
            for a in range(0, 4):
                utilValues[s] = self.valueFunctionRec(old_state, a, 0)

        return utilValues

    def valueFunctionRec(self, old_state: int,  action: int, next_state: int):
        if next_state >= 15:
            return 0
        else:
            return self.π[old_state, action] * self.mdp.stateTransitionModel[old_state, action, next_state] \
            * self.mdp.rewardPerState[next_state] + (self.γ * self.valueFunctionRec(old_state, action, next_state +1))

    def improve(self):
        pass
