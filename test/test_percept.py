import unittest

from src.model.percept import Percept


class TestPercept(unittest.TestCase):
    def test_percept(self):
        percept: Percept = Percept(old_state=1, reward=0, action=2, next_state=2, finished=False, prob=0.33)
        self.assertEqual(percept.oldState, 1)
        self.assertEqual(percept.reward, 0)
        self.assertEqual(percept.action, 2)
        self.assertEqual(percept.nextState, 2)
        self.assertFalse(percept.finished)
        self.assertEqual(percept.prob, 0.33)
