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
        self.ξ = 0  # TODO: fijnheid bepalen

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.actions)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    # Value iteration
    def evaluate(self, percept: Percept):
        Δ = 5000000
        rmax = max(self.mdp.rewardPerState)
        while Δ > 0.1:  # TODO: vervangen door benaderings factor
            Δ = 0
            for state in range(0, 16):
                oldValues = self.v
                self.v = self.valueFunction()
                delta = max(oldValues - self.v)

    def valueFunction(self):
        # TODO: implement bellman funcion
        utilValues = np.zeros(16)
        for i in range(0, 16):
            pass

    def improve(self):
        pass
