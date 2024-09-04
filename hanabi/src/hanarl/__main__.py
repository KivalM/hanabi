from .configs import Config
from argparse import ArgumentParser
from .dqn.train import train_dqn


def args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--wandb_project', type=str, default='hanabi')


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
