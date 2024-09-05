from pettingzoo.classic import hanabi_v5 as hanabi
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from torchrl.envs import PettingZooEnv
import numpy as np

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
        vdn,
    ):
    return HanabiEnv(
        players=players,
        colors=colors,
        ranks=ranks,
        hand_size=hand_size,
        max_information_tokens=max_information_tokens,
        max_life_tokens=max_life_tokens,
        observation_type=observation_type,
        sad=sad,
        shuffle_colors=shuffle_colors,
        vdn=vdn
    )

class TorchHanabiEnv(PettingZooEnv):
    def __init__(
            self, 
            sad, 
            shuffle_colors, 
            players, 
            colors, 
            ranks, 
            hand_size, 
            max_information_tokens, 
            max_life_tokens, 
            observation_type, 
            n_agents,
        ):
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
        self.n_agents = n_agents
        
        self.last_greed_action = None

    def step(self, action, greedy_action):
        if self.sad:
            self.last_greed_action = encode_action_unary(greedy_action, self.out_dim)
        return super().step(action)

    def render(self, mode='human'):
        pass

    def last(self):
        last = super().last()
        print(self.rewards, last[2])
        print(last[1])
        if self.vdn:
            done = last[2]
            current_obs = last[0]['observation']
            current_action = last[0]['action_mask']
            if not done:
                other_obs = self.env.observe(get_other_player(self.agent_selection))
            else:
                other_obs = np.zeros_like(current_obs)
            assert last[0] == self.env.observe(self.agent_selection)
            last[0]['observation'] = (current_obs, other_obs)
            last[0]['action_mask'] = (current_action, current_action)
        
        if self.sad:
            last['greedy_action'] = self.last_greed_action
        return last
    
    def reset(self, seed=None):
        self.last_greed_action = None
        return super().reset(seed)

class HanabiEnv(BaseWrapper):
    def __init__(self, sad, shuffle_colors, players, colors, ranks, hand_size, max_information_tokens, max_life_tokens, observation_type, seed, vdn):
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
        self.vdn = vdn
        
        self.last_greed_action = None

    def step(self, action, greedy_action):
        if self.sad:
            self.last_greed_action = encode_action_unary(greedy_action, self.out_dim)
        return super().step(action)

    def render(self, mode='human'):
        pass

    def last(self):
        last = super().last()
        print(self.rewards, last[2])
        print(last[1])
        if self.vdn:
            done = last[2]
            current_obs = last[0]['observation']
            current_action = last[0]['action_mask']
            if not done:
                other_obs = self.env.observe(get_other_player(self.agent_selection))
            else:
                other_obs = np.zeros_like(current_obs)
            assert last[0] == self.env.observe(self.agent_selection)
            last[0]['observation'] = (current_obs, other_obs)
            last[0]['action_mask'] = (current_action, current_action)
        
        if self.sad:
            last['greedy_action'] = self.last_greed_action
        return last
    
    def reset(self, seed=None):
        self.last_greed_action = None
        return super().reset(seed)

    
def encode_action_unary(action, num_actions):
    return [1 if action == i else 0 for i in range(num_actions)]

def get_other_player(player):
    if player == "player_0":
        return "player_1"
    else:
        return "player_0"