from .configs import *
from argparse import ArgumentParser
from .dqn.train import train_dqn


def args():
    parser = ArgumentParser()
    
    # allow it to override the config
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--wandb_project', type=str, default='Hanabi-RL')

    parser.add_argument('--config', type=str, help='DQN/Rainbow/RDQN(+sad/+op)', choices=['DQN', 'Rainbow', 'RDQN', 'DQN+op', 'DQN+sad'], required=True)
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    args.pop('config')
    print('using config:', config)
    print('args:', args)
    if config=="DQN" or config=="DQN+op" or config=="DQN+sad":
        config = DQNConfig(**args)
    elif config=="Rainbow":
        config = RainbowConfig(**args)

    elif config=="RDQN":
        config = RDQNConfig(**args)
    else:
        raise ValueError(f"Invalid config {config}")
    
    config.wandb = True if args['wandb'] == 1 else False
    print(f"Config: {config}")
    input("Press Enter to continue...")

    return config


def train(config):
    train_dqn(
        config
    )

def evaluate():
    config = Config()
    print(config)


def main():
    config = Config()
    print(config)

if __name__ == '__main__':  
    config = args()
    train(config)
