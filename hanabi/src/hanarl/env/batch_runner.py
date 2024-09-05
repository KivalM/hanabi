from .environment import HanabiEnv
from ..dqn.agent import DQNAgent
from typing import Callable
import  torch.multiprocessing as mp
from collections import deque


class EpisodeBuffer():
    def __init__(self, gamma:float, max_sequence_length:int, multi_step:int, batch_size=1, num_players=2):
        self.multi_step = multi_step
        self.batch_size = batch_size
        self.buffer = []
        self.num_players = num_players
    
    def clear(self):
        self.buffer = []

    def append(self, transition):
        assert not self.can_pop()
        self.buffer.append(transition)

    def can_pop(self):
        return len(self.buffer) >= self.multi_step
    
    def pop(self):
        assert self.can_pop()

        # calculate the discounted reward
        discounted_reward = 0
        for i in range(self.multi_step):
            discounted_reward += self.buffer[i]['reward'] * (self.gamma ** i)

        # create the multi-step transition
        multi_step_transition = {
            'observation': self.buffer[0]['observation'],
            'action': self.buffer[0]['action'],
            'reward': discounted_reward,
            'next_observation': self.buffer[self.multi_step - 1]['next_observation'],
            'done': self.buffer[self.multi_step - 1]['done']
        }

        return self.buffer.popleft()



class BatchRunner():
    def __init__(
            self, 
            env_fn:  Callable[[], HanabiEnv],
            sequence_length:int = 10,
            num_threads:int = mp.cpu_count() - 1,
            num_players:int = 2 # 1 for iql and 2 for vdn
        ):
        self.env_fn = env_fn
        self.sequence_length = sequence_length
        self.num_threads = num_threads
        self.num_players = num_players

    def run_episode(
            self, 
            agent:DQNAgent, 
            epsilon:float, 
            seed:int, 
            env:HanabiEnv
        ):
        env.reset(seed=seed)
        

    def run(self, num_episodes):
        for i in range(num_episodes):
            self.env.reset()
            done = False
            while not done:
                action = self.agent.act(self.env.state)
                _, _, done, _ = self.env.step(action)
        return self.env.get_total_reward()