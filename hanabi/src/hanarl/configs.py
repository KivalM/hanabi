from dataclasses import dataclass
import torch

@dataclass
class Config:
    # General
    seed: int = 999
    wandb: bool = False
    wandb_project: str = 'hanabi'

    # environment
    players: int = 2
    colors: int = 2
    ranks: int = 5
    hand_size: int = 2
    max_information_tokens: int = 3
    train_max_life_tokens: int = 1
    observation_type: str = 'card_knowledge'
    encode_last_action: bool = False
    shuffle_colors: bool = False

    # Agent
    hidden_dim: int = 512
    depth: int = 4
    noisy: bool = True
    distributional: bool = False
    n_atoms: int = 51
    v_min: int = 0
    v_max: int = 20

    dueling: bool = True
    vdn: bool = False
    multi_step: int = 1
    gamma: float = 0.99
    tau: float = 1 # hard update
    double: bool = True




    # Training
    num_epochs: int = 100
    epoch_length: int = 10
    update_target: int = 5
    lr: float = 6.25e-5
    clip_grad: float = 5
    start_eps: float = 1.0
    end_eps: float = 0.01
    eps_decay: float = 0.99

    # Replay Buffer
    prioritized: bool = True
    alpha: float = 0.9
    beta: float = 0.6
    buffer_size: int = 200000
    batch_size: int = 128
    burn_in: int = 10_000

    # Evaluation
    num_eps: int = 50
    eval_eps: int = 0

    # Save
    save_interval: int = 10
    save_dir: str = './models/'

    # Debug
    debug: bool = False
    debug_interval: int = 10
    debug_dir: str = './debug/'

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

