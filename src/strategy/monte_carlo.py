from src.strategy.n_step_q_learning import NstepQlearning


class MonteCarlo(NstepQlearning):
    def __init__(self, env, episode_count):
        super().__init__(env, episode_count)
        self.n = self.episode_count

