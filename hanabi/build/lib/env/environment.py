from pettingzoo.classic import hanabi_v5
from torchrl.envs import PettingZooEnv
from torchrl.envs import TransformedEnv
from torchrl.envs import StepCounter

def make_env_maker(
        version='small', # small, full
        device='cpu'
):   
    '''
    This function creates an environment maker with a specific configuration.
    '''

    def env_fn():
        scenario_name = "hanabi_v5"

        if version == 'small':
            players=2,
            colors=2,
            ranks=5,
            hand_size=2,
            max_information_tokens=3,
            max_life_tokens=1,
        elif version == 'full':
            players=2,
            colors=5,
            ranks=5,
            hand_size=5,
            max_information_tokens=8,
            max_life_tokens=3,
        else:
            raise ValueError(f'Unknown version: {version}, choose from [small, full]')
        
        base = PettingZooEnv(
            task=scenario_name,
            parallel=False,
            use_mask=True,
            players=players,
            ranks=ranks,
            colors=colors,
            hand_size=hand_size,
            max_life_tokens=max_life_tokens,
            max_information_tokens=max_information_tokens,
            categorical_actions=True,
            done_on_any=True,
            device=device,
        )

        env = TransformedEnv(base)
        return env

    return env_fn