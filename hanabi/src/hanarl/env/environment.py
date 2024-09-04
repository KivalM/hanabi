from pettingzoo.classic.hanabi import hanabi
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

def make_env(
        seed,
        players,
        colors,
        ranks,
        hand_size,
        max_information_tokens,
        max_life_tokens,
        observation_type,
        sad,
        shuffle_colors,
        
    ):
    return HanabiEnv(
        seed=seed,
        players=players,
        colors=colors,
        ranks=ranks,
        hand_size=hand_size,
        max_information_tokens=max_information_tokens,
        max_life_tokens=max_life_tokens,
        observation_type=observation_type,
        sad=sad,
        shuffle_colors=shuffle_colors
    )

class HanabiEnv(BaseWrapper):
    def __init__(self, sad, shuffle_colors, players, colors, ranks, hand_size, max_information_tokens, max_life_tokens, observation_type, seed):
        env = hanabi.env(
            players=players,
            colors=colors,
            ranks=ranks,
            hand_size=hand_size,
            max_information_tokens=max_information_tokens,
            max_life_tokens=max_life_tokens,
            observation_type=observation_type,
        )
        super().__init__(env)

        self.sad = sad
        self.shuffle_colors = shuffle_colors

        self.in_dim = self.observation_vector_dim[0]
        self.out_dim = self.action_space('player_0').n

        
        self.last_greed_action = None

    def step(self, action, greedy_action):
        if self.sad:
            self.last_greed_action = encode_action_unary(greedy_action, self.out_dim)
        return super().step(action)

    def render(self, mode='human'):
        pass

    def last(self):
        last = super().last()
        if self.sad:
            last['greedy_action'] = self.last_greed_action
        return last
    
def encode_action_unary(action, num_actions):
    return [1 if action == i else 0 for i in range(num_actions)]



