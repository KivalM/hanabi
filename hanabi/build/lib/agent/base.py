import torch
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    '''
    This class represents the DQN agent.
    '''
    def __init__():
        pass

    @abstractmethod
    def act(self, state:TensorDict):
        '''
        This function performs an action.
        :param state: The state of the environment.
        :return: The action to take.
        '''
        pass
    
    @abstractmethod
    def step(self, transition:TensorDict, training=True):
        '''
        This function stores the transition and updates the model.
        :param transition: The transition to store.
        :param training: Whether to update the model.
        '''
        pass
    
    @abstractmethod
    def save_episode_results(self, max_score, total_score):
        '''
        This function saves the episode.
        :param episode: The episode to save.
        '''
        pass