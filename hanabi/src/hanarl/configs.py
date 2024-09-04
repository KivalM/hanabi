from dataclasses import dataclass
import torch

@dataclass
class Config:
    # General
    seed: int = 42
    wandb: bool = False
    wandb_project: str = 'hanabi'

    # environment
    players: int = 2
    colors: int = 2
    ranks: int = 5
    hand_size: int = 2
    max_information_tokens: int = 3
    train_max_life_tokens: int = 10
    observation_type: str = 'card_knowledge'
    encode_last_action: bool = False
    shuffle_colors: bool = False

    # Agent
    hidden_dim: int = 512
    depth: int = 2
    noisy: bool = False
    distributional: bool = False
    n_atoms: int = 51
    v_min: int = 0
    v_max: int = 20

    dueling: bool = False
    vdn: bool = False
    multi_step: int = 1
    gamma: float = 0.99
    tau: float = 1 # hard update
    double: bool = True




    # Training
    num_epochs: int = 100
    epoch_length: int = 10
    update_target: int = 10
    lr: float = 0.001
    clip_grad: float = 50
    start_eps: float = 0.4
    end_eps: float = 0.02
    eps_decay: float = 0.99

    # Replay Buffer
    prioritized: bool = True
    alpha: float = 0.6
    beta: float = 0.4
    buffer_size: int = 100000
    batch_size: int = 256

    # Evaluation
    num_eps: int = 100
    eval_eps: int = 0
    eval_max_life_tokens: int = 10

    # Save
    save_interval: int = 10
    save_dir: str = './models/'

    # Debug
    debug: bool = False
    debug_interval: int = 10
    debug_dir: str = './debug/'

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

