from pettingzoo.classic.hanabi import hanabi

def make_env_maker(
        version='small', # small, full
        device='cpu'
):   
    '''
    This function creates an environment maker with a specific configuration.
    '''

    def env_fn():

        if version == 'small':
            players=2
            colors=2
            ranks=5
            hand_size=2
            max_information_tokens=3
            max_life_tokens=2
            observation_type = 'card_knowledge'

        elif version == 'full':
            players=2
            colors=5
            ranks=5
            hand_size=5
            max_information_tokens=8
            max_life_tokens=3
            observation_type= 'card_knowledge'
        
        else:
            raise ValueError(f'Unknown version: {version}, choose from [small, full]')
        
        env = hanabi.env(
            players=players,
            colors=colors,
            ranks=ranks,
            hand_size=hand_size,
            max_information_tokens=max_information_tokens,
            max_life_tokens=max_life_tokens,
            observation_type=observation_type,
        )

        return env

    return env_fn