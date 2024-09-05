from torchrl.data.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer
from .agent import DQNAgent
from ..env.environment import HanabiEnv
from ..env.environment import make_env
from ..env.run import collect_data, evaluate
from torchrl.data.replay_buffers.storages import LazyTensorStorage, ListStorage
import wandb
import torch
import datetime
from tqdm import trange
from ..configs import Config
from ..utils import set_all_seeds

def identity(x):
    return x

def train_dqn(
    config:Config
):
    set_all_seeds(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device != 'cuda':
        print('WARNING: Using CPU, this will be slow!')
        input('Press Enter to continue...')

    # Initialize the environment
    env = make_env(
        config.seed,
        config.players,
        config.colors,
        config.ranks,
        config.hand_size,
        config.max_information_tokens,
        config.train_max_life_tokens,
        config.observation_type,
        config.encode_last_action,
        config.shuffle_colors,
        config.vdn
    )

    env.reset(config.seed)
    print(env.in_dim)
    print(env.out_dim)

    # Initialize the replay buffer
    agent = DQNAgent(
        env.in_dim,
        env.out_dim,
        config.hidden_dim,
        config.depth,
        config.noisy,
        config.distributional,
        config.n_atoms,
        config.v_min,
        config.v_max,
        config.dueling,
        config.vdn,
        config.multi_step,
        config.gamma,
        config.tau,
        config.double,
        device=device
    )

    if config.prioritized:
        buffer = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(config.buffer_size),
            alpha=config.alpha,
            beta=config.beta,
            collate_fn=identity
        )
    else:
        buffer = ReplayBuffer(
            storage=ListStorage(config.buffer_size)
        )

    optimizer = torch.optim.AdamW(agent.policy.parameters(), lr=config.lr)

    # burn in replay buffer
    collect_data(env, agent, buffer, 1.0, config.seed, config.burn_in, config.multi_step)

    if config.wandb:
        wandb.init(project='hanabi', config=config)
        wandb.watch(agent.policy)
    
    eps = config.start_eps
    # linear decay
    eps_decay = (config.start_eps - config.end_eps) / (config.num_epochs * config.epoch_length)
    target_update = 0
    for epoch in trange(config.num_epochs):
        for batch_idx in trange(config.epoch_length):
            start_time = datetime.datetime.now()

            # check if we need to update the target network
            num_updates = epoch * config.epoch_length + batch_idx

            if num_updates % config.update_target == 0:
                agent.update_target()
                target_update += 1
            # collect data
            seed = (config.seed + epoch * config.epoch_length + batch_idx) % 7777777
            collect_data(env, agent, buffer, eps, seed, config.batch_size * 2, config.multi_step)

            # sample a batch
            batch, info = buffer.sample(config.batch_size, return_info=True)
            res = agent.compute_loss_and_priority(batch.to(device).detach())
            priority = res['priority']
            loss = res['loss']

            buffer.update_priority(info['index'], priority)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), config.clip_grad)
            optimizer.step()

            # decrease epsilon
            eps = max(config.end_eps, eps - eps_decay)
        
            end_time = datetime.datetime.now()
            print(f'Epoch: {epoch}, Loss: {loss.detach().item()}, Time: {end_time - start_time} Target Update: {target_update}', eps)

        eval_seed = (9917 + epoch * 99999999) % 7777777
        eval_env = make_env(
            eval_seed,
            config.players,
            config.colors,
            config.ranks,
            config.hand_size,
            config.max_information_tokens,
            config.train_max_life_tokens,
            config.observation_type,
            config.encode_last_action,
            config.shuffle_colors
        )
        eval_results = evaluate(eval_env, agent, config.eval_eps,  eval_seed, config.num_eps)

        # if epoch % 10 == 0:
        #     print(f'Epoch: {epoch}, Loss: {loss.detach().item()}, Eval Results: {eval_results}, Time: {end_time - start_time}')
        #     torch.save(agent, f'./models/' + agent.name + f'_epoch_{epoch}.pt')
        #     wandb.save(f'./models/' + agent.name + f'_epoch_{epoch}.pt')

        epoch_results = {
            'epoch': epoch,
            'loss': loss.detach().item(),
            'eval_results': eval_results,
            'time': end_time - start_time
        }
        print(epoch_results)



def train_dqn_(
    env:HanabiEnv,
    agent: DQNAgent,
    optimizer:torch.optim.Optimizer,
    buffer:ReplayBuffer,
    num_epochs:int,
    epoch_length:int,
    update_target:int,
    batch_size:int,
    gamma:float,
):
    """Train a DQN agent on the given environment."""
    for epoch in trange(num_epochs):
        for batch_idx in trange(epoch_length):
            start_time = datetime.datetime.now()

            # check if we need to update the target network
            num_updates = epoch * epoch_length + batch_idx

            if num_updates % update_target == 0:
                agent.update_target()

            # collect data
            data = collect_data(env, agent, buffer, gamma)
            buffer.extend(data)

            # sample a batch
            batch, info = buffer.sample(batch_size, return_info=True)

            res = agent.compute_loss_and_priority(batch)
            priority = res['priority']
            loss = res['loss']

            buffer.update_priority(info['index'], priority)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 50)
            optimizer.step()
        
            end_time = datetime.datetime.now()
        
        eval_seed = (9917 + epoch * 99999999) % 7777777
        eval_results = evaluate(env, agent, 0,  eval_seed)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.detach().item()}, Eval Results: {eval_results}, Time: {end_time - start_time}')
            torch.save(agent, f'./models/' + agent.name + f'_epoch_{epoch}.pt')
            wandb.save(f'./models/' + agent.name + f'_epoch_{epoch}.pt')

        epoch_results = {
            'epoch': epoch,
            'loss': loss.detach().item(),
            'eval_results': eval_results,
            'time': end_time - start_time
        }
        wandb.log(epoch_results)








