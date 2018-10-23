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
        self.policy = np.full((16, 4), 0.25)
        self.ξ = 0 # todo

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.actions)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    # Value iteration
    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        s = percept.oldState
        sn = percept.nextState
        a = percept.action
        r = percept.reward

        # update v-values to match max-q values
        for i in self.v:
            self.v[i] = self.q[i, 0:4].max()

    def improve(self):
        pass
        #for i in self.v:
            # self.policy[]

