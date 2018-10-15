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
