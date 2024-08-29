from ..agent.base import BaseAgent
from torchrl.envs import TransformedEnv
from typing import List, Dict, Callable
from collections import namedtuple
import time
from dataclasses import dataclass, asdict
from tqdm.notebook import trange, tqdm_notebook
import numpy as np
from tensordict import TensorDict
import wandb

def run_episode_single_agent(
        env:TransformedEnv,
        agent:BaseAgent,
    ) :
    """Runs a single episode of the game with the given agent.

    Args:
        agents: A dictionary of agents to run the episode with. The key should match the player id.
        env: The environment to run the episode in.

    Returns:
        episode_returns: A named tuple containing the returns and the number of steps in the episode.
    """
    rewards = {
        agent: 0.0
        for agent in env.agents
    }

    max_rewards = rewards.copy()
    # iterate through the environment until the episode is over
    state = env.reset()
    for player in env.agent_iter():

        state = state[player]

        if state['done']:
            action = None
            break
        
        action = agent.act(state)

        env_action = TensorDict({agent: {"action": [action]}})

        next = env.step(env_action)['next']
        rewards[player] += next[player]['reward']
        max_rewards[player] = max(max_rewards[player], rewards[player])
        transition = TensorDict({
            "state": {
                'observation': state[player, 'observation', 'observation'],
                'action_mask': state[player, 'action_mask'],
                'greedy_action': None,
            },
            "action": action,
            "reward": next[player]['reward'],
            "next_state":{
                'observation': next[player, 'observation', 'observation'],
                'action_mask': next[player, 'action_mask'],
                'greedy_action': None,
            },
            "done": next['done'],
        })

        agent.step(transition)

        state = next

    agent.save_episode_results(max(max_rewards.values()), max(rewards.values()))
