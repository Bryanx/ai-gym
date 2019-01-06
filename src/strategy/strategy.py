from gym import Env
import random

from src.model.mdp import MarkovDecisionProcess
import numpy as np
from math import e

from src.model.percept import Percept


class Strategy:
    def __init__(self, env: Env, episode_count: int):
        self.state_count = env.observation_space.n
        self.action_count = env.action_space.n
        self.episode_count = episode_count
        self.env = env
        self.mdp = MarkovDecisionProcess(self.state_count, self.action_count)
        self.q = np.zeros(
            (self.state_count, self.action_count))  # q-values: for each state-action the average reward to the goal.
        self.α = 0.1  # learning rate, determines to what extent new info overrides old info.
        self.v = np.zeros(self.state_count)  # v-values: this will decide the policy - the max q value of each state.
        self.γ = 0.9  # discount factor -> determines the importance of future rewards.
        self.λ = 0.7  # this value causes exponential decay on the exploration factor

        # attributes for policy improvement
        # in the initial policy each action has the same chance of being executed
        self.π = np.full((self.state_count, self.action_count), 1 / self.action_count)
        self.ε = 1  # exploration factor
        self.εmin = 0.1  # the min exploration factor
        self.εmax = 1  # the max exploration factor
        self.t = 0  # time factor

    # TODO: fix
    def next_action(self, state):
        # get next action from policy here
        # return np.random.choice(np.arange(self.action_count), 1, p=self.π[state])[0]
        return self.pick_choice(state)
        # return np.random.choice(self.mdp.A)

    def pick_choice(self, state):
        selection = list()
        policy_of_state = self.π[state]

        for a in self.mdp.A:
            number_of_times = int(round(policy_of_state[a] * 100))
            for j in range(number_of_times):
                selection.append(a)

        return np.random.choice(selection)

    def learn(self, percept: Percept):
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

    # if 2 q values are max a random choice is made.
    def get_best_action_for_state(self, s: int):
        return np.random.choice(np.flatnonzero(self.q[s] == self.q[s].max()))
