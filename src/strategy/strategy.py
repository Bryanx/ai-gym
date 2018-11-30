from gym import Env

from src.model.mdp import MarkovDecisionProcess
import numpy as np
import random
from math import e

from src.model.percept import Percept

class Strategy:
    def __init__(self, env: Env, episode_count: int):
        self.state_count = env.observation_space.n
        self.action_count = env.action_space.n
        self.episode_count = episode_count
        self.mdp = MarkovDecisionProcess(self.state_count, self.action_count)
        self.q = np.zeros((self.state_count, self.action_count))  # q-values: for each state-action the average reward to the goal.
        self.α = 0.1  # learning rate, determines to what extent new info overrides old info.
        self.v = np.zeros(self.state_count)  # v-values: this will decide the policy - the max q value of each state.
        self.γ = 0.9  # discount factor -> determines the importance of future rewards.
        self.λ = 0.7  # this value causes exponential decay on the exploration factor

        # attributes for policy improvement
        # in the initial policy each action has the same chance of being executed
        self.π = np.full((self.state_count, self.action_count), 1/self.action_count)
        self.ε = 1  # exploration factor
        self.εmin = 0.1  # the min exploration factor
        self.εmax = 1  # the max exploration factor
        self.t = 0  # time factor

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.A)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    def evaluate(self, percept: Percept):
        raise NotImplementedError("Please Implement this method")

    def improve(self):
        εmax, εmin, q, α, λ, t, S, A = self.εmax, self.εmin, self.q, self.α, self.λ, self.t, self.mdp.S, self.mdp.A
        self.t += 1
        for s in S:
            best_a = self.get_best_action_for_state(s)
            for a in A:
                if a == best_a:
                    self.π[s, a] = 1 - self.ε + (self.ε / len(A))
                else:
                    self.π[s, a] = self.ε / len(A)
            self.ε = εmin + (εmax - εmin) * e ** (-λ * t)

    def get_best_action_for_state(self, s: int):
        return np.argmax([self.q[s, a] for a in self.mdp.A])
