from dataclasses import dataclass, asdict
from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer, LazyTensorStorage
from .net import DQN
from .agent import DQNAgent

from ...env.play import run_episode_single_agent
from ...env.environment import make_env_maker
import torch

@dataclass
class TrainConfig:
    # QNetConfig
    state_dim:int
    action_dim:int
    device:str = 'cuda'
    hidden_dim:int = 512
    depth:int = 3
    noisy:bool = True 
    distributional:bool = False
    duel:bool = True

    # MemoryConfig
    capacity:int = 200_000
    batch_size:int = 256
    prioritized:bool = True

    # DQNAgentConfig
    double:bool = True
    tau:float = 0.01
    gamma:float = 0.99
    policy_update_freq:int = 5
    target_update_freq:int = 20
    
    lr: float = 0.0005

    # ExplorationConfig
    start_epsilon:float = 1
    end_epsilon:float = 0.01

    # TrainingConfig
    n_times:int = 1000*1000

    # The additional parameters
    sad:bool = False
    shuffle_observation:bool = False

    # checkpoint frequency
    checkpoint_freq:int = 50000
    checkpoint_dir:str = 'checkpoints'

    # logging
    log:bool = True
    log_interval:int = 10_000

def identity(x):
    return x

class DQNTrainer:
    def __init__(self, config:TrainConfig, env_maker=make_env_maker):
        self.config = config

        # env
        self.env_maker = env_maker('small', "cuda")
        env = self.env_maker()
        config.state_dim = env.observation_vector_dim[0]
        config.action_dim = env.action_space('player_0').n

        # DQN
        self.policy_net = DQN(config.state_dim, config.action_dim, config.hidden_dim, config.depth, config.noisy, config.distributional, config.duel)
        self.policy_net.to(config.device)
        # double DQN
        if config.double:
            self.target_net = DQN(config.state_dim, config.action_dim, config.hidden_dim, config.depth, config.noisy, config.distributional, config.duel)
            self.target_net.to(config.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        
        if config.prioritized:
            self.buffer = PrioritizedReplayBuffer(alpha=0.5, beta=0.5, storage=LazyTensorStorage(config.capacity, device='cuda'), batch_size=config.batch_size, collate_fn=identity, eps=1e-6)
        else:
            self.buffer = ReplayBuffer(storage=LazyTensorStorage(config.capacity, device='cuda'), batch_size=config.batch_size)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters())

        # Agent
        self.agent = DQNAgent(self.policy_net, self.target_net, self.optimizer, self.buffer, config.double, config.tau, config.gamma, config.policy_update_freq, config.target_update_freq, config.start_epsilon, config.end_epsilon, config.n_times, config.sad, config.shuffle_observation, config.checkpoint_freq, config.checkpoint_dir, config.log, config.log_interval, config.device)

    def train(self):
        env = self.env_maker()
        while not self.agent.done:
            env = self.env_maker()
            run_episode_single_agent(env, self.agent)


    def test(self):
        pass

    def save(self, path:str):
        pass

    def load(self, path:str):
        pass

    def __str__(self):
        return str(asdict(self.config))

    def __repr__(self):
        return str(asdict(self.config))