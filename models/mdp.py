import numpy as np


class MarkovDecisionProcess:
    def __init__(self, amount_of_states: int, amount_of_actions: int):
        states = np.arange(0, amount_of_states, 1)
        self.rewardPerState = np.zeros(amount_of_states)
        self.actions = np.arange(0, amount_of_actions, 1)
        self.stateTransitionModel = np.zeros((amount_of_states, amount_of_actions, amount_of_states))

    def update(self, percept):
        self.stateTransitionModel[percept.oldState][percept.action][percept.nextState] = round(percept.prob, 2)
        self.rewardPerState[percept.nextState] = percept.reward
        print(3)

    def __str__(self) -> str:
        return "Actions: \n" + str(self.stateTransitionModel) + "\nRewardPerState:\n" + str(self.rewardPerState)
