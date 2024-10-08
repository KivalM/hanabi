from dataclasses import dataclass
import torch

@dataclass
class BaseConfig:
    # General
    seed: int = 99
    wandb: bool = True
    wandb_project: str = 'hanabi'

    # environment
    players: int = 2
    colors: int = 2
    ranks: int = 5
    hand_size: int = 2
    max_information_tokens: int = 3
    train_max_life_tokens: int = 2
    observation_type: str = 'card_knowledge'
    encode_last_action: bool = False
    shuffle_colors: bool = False
    step_reward_lower_bound: float = None # a penalty for an illegal move
    hint_reward: float = 0.0 # a reward for giving a hint
    discard_reward: float = 0.0 # a reward for discarding a card

    # Agent
    hidden_dim: int = 512
    depth: int = 4
    noisy: bool = True
    noise_std: float = 0.05
    distributional: bool = True
    n_atoms: int = 51
    v_min: int = -100
    v_max: int = 100

    dueling: bool = True
    vdn: bool = False
    multi_step: int = 3
    max_seq_len = 1
    gamma: float = 0.99
    tau: float = 1 # hard update
    # tau: float = 0.005 # soft update
    double: bool = True

    # Training
    num_epochs: int = 100
    epoch_length: int = 100
    policy_update: int = 256 # number of game steps before policy update
    update_target: int = 100 # number of policy updates before target update
    lr: float = 6.25e-5
    optimizer_eps: float = 1e-8
    clip_grad: float = 100.0
    start_eps: float = 0.0
    end_eps: float = 0.0
    burn_in_eps: int = 1

    # Replay Buffer
    prioritized: bool = True
    # lower alpha means more prioritization i.e more weight to TD error
    alpha: float = 0.9
    # lower beta means more importance sampling
    beta: float = 0.6
    buffer_size: int = 10_000
    batch_size: int = 512
    burn_in: int = 512

    # Evaluation
    num_eps: int = 100
    eval_eps: int = 0

    save_dir: str = './models/'
       # debug
    debug: bool = False

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
