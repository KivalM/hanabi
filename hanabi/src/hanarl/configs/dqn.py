from . import BaseConfig
import torch
from dataclasses import dataclass

@dataclass
class DQNConfig(BaseConfig):
   # General
   seed: int = 99
   wandb: bool = True
   wandb_project: str = 'hanabi'


   # Agent
   hidden_dim: int = 512
   depth: int = 2
   noisy: bool = False
   distributional: bool = False
   dueling: bool = False
   vdn: bool = False
   multi_step: int = 0
   max_seq_len = 1
   gamma: float = 0.99
   # tau: float = 1 # hard update
   tau: float = 0.005 # soft update
   double: bool = True

   # Training
   num_epochs: int = 120
   epoch_length: int = 1000
   policy_update: int = 50 # number of game steps before policy update
   update_target: int = 1 # number of policy updates before target update
   lr: float = 1e-4
   optimizer_eps: float = 1e-8
   clip_grad: float = 50.0
   start_eps: float = 1.0
   end_eps: float = 0.0
   burn_in_eps: int = 1

   # Replay Buffer
   prioritized: bool = False
   # lower alpha means more prioritization i.e more weight to TD error
   alpha: float = 0.6
    # lower beta means more importance sampling
   beta: float = 0.4
   buffer_size: int = 10_000
   batch_size: int = 256
   burn_in: int = 1_000

   # Evaluation
   num_eps: int = 100
   eval_eps: int = 0

   save_dir: str = './models/'
   # debug
   debug: bool = False

   # Device
   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'