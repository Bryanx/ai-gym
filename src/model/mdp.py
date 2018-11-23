import numpy as np


class MarkovDecisionProcess:
    def __init__(self, state_count: int, action_count: int):
        self.rewardPerState = np.zeros(state_count)
        self.actions = np.arange(0, action_count, 1)
        self.stateTransitionModel = np.zeros((state_count, action_count, state_count))

    def update(self, percept):
        self.stateTransitionModel[percept.oldState][percept.action][percept.nextState] = round(percept.prob, 2)
        self.rewardPerState[percept.nextState] = percept.reward

    def __str__(self) -> str:
        return "Actions: \n" + str(self.stateTransitionModel) + "\nRewardPerState:\n" + str(self.rewardPerState)
