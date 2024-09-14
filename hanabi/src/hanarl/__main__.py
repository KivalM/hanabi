from .configs.dqn import DQNConfig
from .configs.rainbow import RainbowConfig
from argparse import ArgumentParser
from .dqn.train import train_dqn


def args():
    parser = ArgumentParser()
    
    # allow it to override the config
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--wandb_project', type=str, default='Hanabi-RL')
    parser.add_argument('--observation_type', type=str, choices=['card_knowledge', 'seer'])
    parser.add_argument('--encode_last_action', type=int)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--config', type=str, help='DQN/Rainbow/RDQN', choices=['DQN', 'Rainbow', 'RDQN'], required=True)
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    args.pop('config')
    print('using config:', config)
    # remove any args that dont have a value

    args = {k: v for k, v in args.items() if v is not None}

    if config=="DQN" :
        config = DQNConfig(**args)
    elif config=="Rainbow":
        config = RainbowConfig(**args)
        assert config.encode_last_action == 0, "Does not make sense to encode last action for Rainbow"

    elif config=="RDQN":
        # config = RDQNConfig(**args)
        pass
    else:
        raise ValueError(f"Invalid config {config}")
    
    config.wandb = True if args['wandb'] == 1 else False
    return config


def train(config):
    train_dqn(
        config
    )

if __name__ == '__main__':  
    config = args()
    train(config)
