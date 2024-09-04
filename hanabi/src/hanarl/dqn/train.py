from torchrl.data.replay_buffers import ReplayBuffer
from .agent import DQNAgent
from ..env.environment import HanabiEnv
import torch
import datetime
import tqdm

def train_dqn(
    env:HanabiEnv,
    agent: DQNAgent,
    buffer:ReplayBuffer,
    num_epochs:int,
    epoch_length:int,
    update_target:int,
    batch_size:int,
    gamma:float,
):
    """Train a DQN agent on the given environment."""
    for epoch in range(num_epochs):
        for batch_idx in range(epoch_length):
            # run a single episode
            num_updates = epoch * epoch_length + batch_idx

            if num_updates % update_target == 0:
                agent.update_target()
            
            # time
            start_time = datetime.datetime.now()

            batch, info = buffer.sample(batch_size, return_info=True)

            loss = agent.compute_loss(batch)
            priority = loss['priority']
            loss = loss['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 50)
