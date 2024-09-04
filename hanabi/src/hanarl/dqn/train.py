from torchrl.data.replay_buffers import ReplayBuffer
from .agent import DQNAgent
from ..env.environment import HanabiEnv
from ..env.run import collect_data, evaluate
import wandb
import torch
import datetime
from tqdm import trange

def train_dqn(
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








