from .configs import Config
from argparse import ArgumentParser
from .dqn.train import train_dqn


def args():
    parser = ArgumentParser()
    
    # allow it to override the config
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--wandb_project', type=str, default='hanabi')
    parser.add_argument('--players', type=int, default=2)
    parser.add_argument('--colors', type=int, default=2)

    args = parser.parse_args()
    config = Config(**vars(args))
    return config


def train():
    config = Config()
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
    
    train()
