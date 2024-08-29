from dataclasses import dataclass, asdict
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage
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
    hidden_dim:int = 256
    depth:int = 2
    noisy:bool = False 
    distributional:bool = False
    duel:bool = False

    # MemoryConfig
    capacity:int = 50_000
    batch_size:int = 200
    prioritized:bool = False

    # DQNAgentConfig
    double:bool = True
    tau:float = 1e-3
    gamma:float = 0.99
    policy_update_freq:int = 200
    target_update_freq:int = 1000

    # ExplorationConfig
    start_epsilon:float = 1.0
    end_epsilon:float = 0.01

    # TrainingConfig
    n_times:int = 1000*100

    # The additional parameters
    sad:bool = False
    shuffle_observation:bool = False

    # checkpoint frequency
    checkpoint_freq:int = 5000


class DQNTrainer:
    def __init__(self, config:TrainConfig, env_maker=make_env_maker):
        self.config = config

        # env
        self.env_maker = env_maker
        env = self.env_maker()
        config.state_dim = env.observation_spec['player_0']['observation']['observation'].shape[-1]
        config.action_dim = env.action_spec['player_0']['action'].space.n

        # DQN
        self.policy_net = DQN(config.state_dim, config.action_dim, config.hidden_dim, config.depth, config.noisy, config.distributional, config.duel)

        # double DQN
        if config.double:
            self.target_net = DQN(config.state_dim, config.action_dim, config.hidden_dim, config.depth, config.noisy, config.distributional, config.duel)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        
        if config.prioritized:
            self.buffer = TensorDictPrioritizedReplayBuffer(alpha=0.6, beta=0.4, storage=LazyTensorStorage(config.capacity), batch_size=config.batch_size)
        else:
            self.buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(config.capacity), batch_size=config.batch_size)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters())

        # Agent
        self.agent = DQNAgent(self.policy_net, self.target_net, self.optimizer, self.buffer, config.double, config.tau, config.gamma, config.policy_update_freq, config.target_update_freq, config.start_epsilon, config.end_epsilon, config.n_times, config.sad, config.shuffle_observation, config.checkpoint_freq)


    def train(self):
        env = self.env_maker()
        while not self.agent.done:
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