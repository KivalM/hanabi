from torchrl.data.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer
from .agent import DQNAgent
from ..env.environment import make_env
from torchrl.data.replay_buffers.storages import LazyTensorStorage, ListStorage
import wandb
import torch
import datetime
from tqdm import trange
from ..configs import BaseConfig
from dataclasses import asdict
from ..utils import set_all_seeds
from ..env.batch_runner import BatchRunner
# time
import datetime
import os

def identity(x):
    return x

def train_dqn(
    config:BaseConfig
):
    set_all_seeds(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device != 'cuda':
        print(f'WARNING: Using {device}, this will be slow!')
        input('Press Enter to continue...')

    # Initialize the environment
    def env_maker():
        return make_env(
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

    runner = BatchRunner(env_maker, config.max_seq_len, config.multi_step, config.players, config.step_reward_lower_bound, config.debug, config.hint_reward, config.discard_reward)

    env = env_maker()
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
        config.noise_std,
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

    if not config.prioritized:
        config.alpha = 0.0
        config.beta = 1.0

    buffer = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(config.buffer_size),
        alpha=config.alpha,
        beta=config.beta,
        collate_fn=identity,
        batch_size=config.batch_size,
        prefetch=3
    )
    optimizer = torch.optim.AdamW(agent.policy.parameters(), lr=config.lr, eps=config.optimizer_eps)



    run_name = f'{agent.name}_{config.seed}'
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + run_name
    # make a folder for experiments
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    
    # make a folder for this run
    assert not os.path.exists(f'{config.save_dir}/{run_name}')
    save_dir = f'{config.save_dir}/{run_name}'
    os.makedirs(save_dir)
    # save the config
    with open(f'{save_dir}/config.txt', 'w') as f:
        f.write(str(asdict(config)))

    if config.debug:
        debug_dir = f'{save_dir}/debug'
        os.makedirs(debug_dir)

    if config.wandb:
        wandb.init(project=config.wandb_project, name=run_name, config=asdict(config))
        wandb.watch(agent.policy)
        # open the wandb page
        wandb_url = f'https://wandb.ai/{config.wandb_project}/{run_name}'
        print(f'Wandb: {wandb_url}')
    
    # burn in replay buffer
    print('Burn in buffer:', config.burn_in, "eps:", config.burn_in_eps)
    # collect_data(env, agent, buffer, 1.0, config.seed, config.burn_in, config.multi_step)
    stats = runner.run(agent, buffer, config.burn_in_eps, config.seed, config.burn_in, config.debug)
    if config.debug:
        stats.log(f'{debug_dir}/burn_in_stats.csv')

    print(f'Buffer size: {len(buffer)}')
    eps = config.start_eps
    # linear decay
    eps_decay = (config.start_eps - config.end_eps) / (config.num_epochs * config.epoch_length)
    target_update = 0
    eval_logs = []
    steps_train = 0
    for epoch in trange(config.num_epochs):
        epoch_start_time = datetime.datetime.now()
        epoch_losses = []
        for batch_idx in trange(config.epoch_length):
            start_time = datetime.datetime.now()

            # check if we need to update the target network
            num_updates = epoch * config.epoch_length + batch_idx

            if num_updates % config.update_target == 0:
                agent.update_target()
                target_update += 1
            # collect data
            seed = (config.seed + epoch * config.epoch_length + batch_idx) * 99999999 % 99999999997
            # collect_data(env, agent, buffer, eps, seed, config.batch_size * 2, config.multi_step)
            stats = runner.run(agent, buffer, eps, seed, config.policy_update, config.debug)
            if config.debug:
                stats.log(f'{debug_dir}/epoch_{epoch}_batch_{batch_idx}_stats.csv')
            
            # sample a batch
            batch = buffer.sample()
            res = agent.compute_loss_and_priority(batch.to(device).detach())
            priority = res['priority']
            loss = res['loss']

            # aggregate priorities
            batch.set('td_error', priority)
            buffer.update_tensordict_priority(batch)
            steps_train += config.batch_size
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), config.clip_grad)
            optimizer.step()

            # decrease epsilon
            eps = max(config.end_eps, eps - eps_decay)
        
            end_time = datetime.datetime.now()
            epoch_losses.append(loss.detach().item())
            print(f'Epoch: {epoch}, Loss: {loss.detach().item()}, Time: {end_time - start_time} Target Update: {target_update}', eps, len(buffer))

        eval_seed = (config.seed + 9917 + epoch * 99999999) % 7777777
        eval_results = runner.evaluate(agent, config.num_eps, eval_seed, config.eval_eps)
        epoch_end_time = datetime.datetime.now()
        epoch_results = {
            'epoch': epoch,
            # 'loss': loss.detach().item(),
            'loss': sum(epoch_losses) / len(epoch_losses),
            'eval_results': eval_results,
            'time:': (epoch_end_time - epoch_start_time).total_seconds(),
            'eps': eps,
            'target_update': target_update,
            'buffer_size': len(buffer),
            "steps_train": steps_train

        }
        print(epoch_results)
        eval_logs.append(epoch_results)

        save_name = f'{save_dir}/epoch_{epoch}.pt'
        agent.save(save_name)

        if config.wandb:
            wandb.save(save_name)
            wandb.log({
                'epoch': epoch,
                # 'loss': loss.detach().item(),
                'loss': sum(epoch_losses) / len(epoch_losses),
                'eval_results': eval_results,
                'time:': (epoch_end_time - epoch_start_time).total_seconds(),
                'eps': eps,
                'target_update': target_update,
                'buffer_size': len(buffer),
                "steps_train": steps_train
            })
    #  final evaluation
    eval_seed = ((config.seed + 9918 + epoch) * 99999999) % 777777777
    eval_results = runner.evaluate(agent, config.num_eps*10, eval_seed, config.eval_eps)
    epoch_end_time = datetime.datetime.now()
    epoch_results = {
        'epoch': epoch,
        'loss': loss.detach().item(),
        'eval_results': eval_results,
        'time:': (epoch_end_time - epoch_start_time).total_seconds(),
        'eps': eps
    }
    print(epoch_results)
    eval_logs.append(epoch_results)

    if config.wandb:
        wandb.log({
            'epoch': epoch,
            'loss': loss.detach().item(),
            'eval_results': eval_results,
            'time': (end_time - start_time).total_seconds(),
            'eps': eps,
            'target_update': target_update,
            'buffer_size': len(buffer),
        })

    # save eval logs
    with open(f'{save_dir}/eval_logs.txt', 'w') as f:
        for log in eval_logs:
            f.write(str(log) + '\n')

# def train_dqn_(
#     env:HanabiEnv,
#     agent: DQNAgent,
#     optimizer:torch.optim.Optimizer,
#     buffer:ReplayBuffer,
#     num_epochs:int,
#     epoch_length:int,
#     update_target:int,
#     batch_size:int,
#     gamma:float,
# ):
#     """Train a DQN agent on the given environment."""
#     for epoch in trange(num_epochs):
#         for batch_idx in trange(epoch_length):
#             start_time = datetime.datetime.now()

#             # check if we need to update the target network
#             num_updates = epoch * epoch_length + batch_idx

#             if num_updates % update_target == 0:
#                 agent.update_target()

#             # collect data
#             data = collect_data(env, agent, buffer, gamma)
#             buffer.extend(data)

#             # sample a batch
#             batch, info = buffer.sample(batch_size, return_info=True)

#             res = agent.compute_loss_and_priority(batch)
#             priority = res['priority']
#             loss = res['loss']

#             buffer.update_priority(info['index'], priority)

#             optimizer.zero_grad()
#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 50)
#             optimizer.step()
        
#             end_time = datetime.datetime.now()
        
#         eval_seed = (9917 + epoch * 99999999) % 7777777
#         eval_results = evaluate(env, agent, 0,  eval_seed)

#         if epoch % 10 == 0:
#             print(f'Epoch: {epoch}, Loss: {loss.detach().item()}, Eval Results: {eval_results}, Time: {end_time - start_time}')
#             torch.save(agent, f'./models/' + agent.name + f'_epoch_{epoch}.pt')
#             wandb.save(f'./models/' + agent.name + f'_epoch_{epoch}.pt')

#         epoch_results = {
#             'epoch': epoch,
#             'loss': loss.detach().item(),
#             'eval_results': eval_results,
#             'time': end_time - start_time
#         }
#         wandb.log(epoch_results)








