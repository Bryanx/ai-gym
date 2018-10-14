import gym
import random
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, gym_name: str, episodes: int):
        self.env = gym.make(gym_name)
        self.episodes = range(episodes)
        self.strategy = Strategy()

    def learn(self):
        for n in self.episodes:
            old_state = self.env.reset()
            print("Episode: " + str(n))
            for t in range(100):
                action = self.strategy.next_action()  # strategy should decide next action here
                new_state, reward, done, info = self.env.step(action)
                percept = Percept(old_state, reward, action, new_state, done, info.get("prob"))
                print(percept)
                self.env.render()
                self.strategy.learn(percept)
                old_state = new_state
                if done:
                    print(f"Episode {n} finished after {t+1} timesteps")
                    break
        self.print()

    def print(self):
        # testing / printing:
        v = self.strategy.v
        x = np.array([v[0:4], v[4:8], v[8:12], v[12:16]])
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='hot', interpolation='nearest')
        for i in range(0, 4):
            for j in range(0, 4):
                ax.text(j, i, round(x[i, j], 4),
                        ha="center", va="center", color="b")
        plt.show()


class MarkovDecisionProcess:
    def __init__(self, amount_of_states: int, amount_of_actions: int):
        states = np.arange(0, amount_of_states, 1)
        self.rewardPerState = np.transpose((states, np.zeros(amount_of_states)))
        self.actions = np.arange(0, amount_of_actions, 1)
        self.stateTransitionModel = np.zeros((amount_of_states, amount_of_actions, amount_of_states))

    def update(self, percept):
        self.stateTransitionModel[percept.oldState][percept.action][percept.nextState] = round(percept.prob, 2)

    def __str__(self) -> str:
        return "Actions: \n" + str(self.stateTransitionModel) + "\nRewardPerState:\n" + str(self.rewardPerState)


class Percept:
    def __init__(self, old_state: int, reward: int, action: int, next_state: int, finished: bool, prob: float):
        self.oldState = old_state
        self.reward = reward
        self.action = action
        self.nextState = next_state
        self.finished = finished
        self.prob = prob

    def __str__(self):
        if self.action == 0:
            action_str = "left"
        elif self.action == 1:
            action_str = "down"
        else:
            action_str = "right"
        return f'From {self.oldState} pressed {action_str}, went to {self.nextState}, got reward {self.reward}'


class Strategy:

    def __init__(self):
        self.mdp = MarkovDecisionProcess(16, 4)
        self.v = np.zeros(16)  # v values -> this will decide the policy - the max q value of each state.
        self.q = np.zeros((16, 4))  # -> q values -> for each state-action the average reward to the goal.
        self.gamma = 0.9  # discount factor -> determines the importance of future rewards.
        self.alpha = 0.1  # learning rate, determines to what extent new info overrides old info.

    def next_action(self):
        # get next action from policy here
        return random.choice(self.mdp.actions)

    def learn(self, percept):
        self.evaluate(percept)
        self.improve()

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        s = percept.oldState
        sn = percept.nextState
        a = percept.action
        r = percept.reward
        # q of current state = reward + max-q of next state    (gamma and alpha tweak the equation)
        self.q[s, a] += self.alpha * (r + self.gamma * (self.q[sn, 0:4].max() - self.q[s, a]))

        # update v-values to match max-q values
        for i in range(0, 16):
            self.v[i] = self.q[i, 0:4].max()

    def improve(self):
        # improve policy here
        pass


agent = Agent('FrozenLake-v0', episodes=1000)
agent.learn()
