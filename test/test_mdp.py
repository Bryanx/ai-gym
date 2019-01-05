import unittest

from src.model.mdp import MarkovDecisionProcess
from src.model.percept import Percept


class TestMDP(unittest.TestCase):
    percept: Percept = Percept(old_state=14, reward=1, action=2, next_state=15, finished=True, prob=0.33)
    mdp: MarkovDecisionProcess = MarkovDecisionProcess(state_count=16, action_count=4)
    mdp.update(percept)

    def test_reward_per_state(self):
        self.assertEqual(self.mdp.rewardPerState[15], 1)

    def test_state_transitioning_model(self):
        self.assertEqual(self.mdp.stateTransitionModel[14][2][15], 0.33)
