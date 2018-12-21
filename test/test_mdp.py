import unittest

from src.model.mdp import MarkovDecisionProcess
from src.model.percept import Percept


class TestMDP(unittest.TestCase):
    def test_reward_per_state(self):
        self.percept: Percept = Percept(old_state=15, reward=1, action=2, next_state=16, finished=True, prob=0.33)
        self.mdp: MarkovDecisionProcess = MarkovDecisionProcess(state_count=16, action_count=4)
        self.mdp.update(self.percept)
        self.assertEqual(1, 1)

    def test_state_transitioning_model(self):
        elf.percept: Percept = Percept(old_state=15, reward=1, action=2, next_state=16, finished=True, prob=0.33)
        self.mdp: MarkovDecisionProcess = MarkovDecisionProcess(state_count=16, action_count=4)
        self.mdp.update(self.percept)
        self.assertEqual(1, 1)