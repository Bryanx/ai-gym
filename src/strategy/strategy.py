from gym import Env

from src.model.mdp import MarkovDecisionProcess
import numpy as np
import random

from src.model.percept import Percept


class Strategy:
    def __init__(self, env: Env, episode_count: int):
        self.state_count = env.observation_space.n
        self.action_count = env.action_space.n
        self.episode_count = episode_count
        self.mdp = MarkovDecisionProcess(self.state_count, self.action_count)
        self.v = np.zeros(self.state_count)  # v values -> this will decide the policy - the max q value of each state.
        self.q = np.zeros((self.state_count, self.action_count))  # -> q values -> for each state-action the average reward to the goal.
        self.γ = 0.9  # discount factor -> determines the importance of future rewards.
        self.α = 0.1  # learning rate, determines to what extent new info overrides old info.
        self.π = np.full((self.state_count, self.action_count), 0.25)
        self.n = 50
        self.p = []
        self.ξ = 0.1  # fineness factor

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.actions)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    def evaluate(self, percept: Percept):
        raise NotImplementedError("Please Implement this method")

    def improve(self):
        raise NotImplementedError("Please Implement this method")
