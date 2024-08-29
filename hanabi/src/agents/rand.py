import numpy as np

class RandomAgent:
    def __init__(self, game):
        self.game = game

    def act(self, state):
        mask = state['action_mask']
        if mask is not None:
            return np.random.choice(np.where(mask == 1)[0])
        else:
            return np.random.choice(self.game.action_space)