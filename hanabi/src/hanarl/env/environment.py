from pettingzoo.classic import hanabi_v5
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
    def __init__(self, sad, shuffle_colors, *args, **kwargs):
        env = hanabi_v5.env(
            players=2,
            colors=2,
            ranks=5,
            hand_size=2,
            max_information_tokens=3,
            max_life_tokens=2
        )
        super().__init__(env)
        self.players=2
        self.colors=2
        self.ranks=5
        self.hand_size=2
        self.max_information_tokens=3
        self.max_life_tokens=2 
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



def collect_data(env, agent, n_times=1000):
    data = []
    for _ in range(n_times):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            data.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return data