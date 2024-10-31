from .environment import HanaEnv
from ..dqn.agent import DQNAgent
from typing import Callable
import  torch.multiprocessing as mp
from tensordict import TensorDict
from typing import List
import torch
import torch.multiprocessing as mp
from torchrl.data.replay_buffers import TensorDictPrioritizedReplayBuffer
import numpy as np
import pandas as pd
from tqdm import trange

class StatsCollector():
    def __init__(self, out_dim:int):
        self.out_dim = out_dim
        self.stats = []
    
    def store_episode(self, episode: List[TensorDict], priorities: List[torch.Tensor], epsilon:float):
        i = 0
        for priority, transition in zip(priorities, episode):
            greedy_action = transition["action"]["greedy_action"].detach().cpu().item()
            action = transition["action"]["action"].detach().cpu().item()
            actual_reward = transition["actual_reward"].detach().cpu().item()
            multi_step_reward = transition["reward"].detach().cpu().item()
            bootstrap = transition["bootstrap"].detach().cpu().item()
            done = transition["done"].detach().cpu().item()
            self.stats.append({
                "i": i,
                "epsilon": epsilon,
                "bootstrap": bootstrap,
                "greedy_action": greedy_action,
                "action": action,
                "actual_reward": actual_reward,
                "multi_step_reward": multi_step_reward,
                "priority": priority.item(),
                "done": done
            })
            i += 1
    
    def log(self, path:str):
        with open(path, 'w') as f:
            df = pd.DataFrame(self.stats)
            df.to_csv(f, index=False)

def zeros_like_td(transition: TensorDict):
    return TensorDict({
            "observation": torch.zeros_like(transition["observation"]),
            "action_mask": torch.zeros_like(transition["action_mask"]),
            "reward": torch.zeros_like(transition["reward"]),
            "done": torch.ones_like(transition["done"]),
            "action": {
                "action": torch.zeros_like(transition["action"]["action"]),
                "greedy_action": torch.zeros_like(transition["action"]["greedy_action"]),
            },
            "next":{
                "observation": torch.zeros_like(transition["observation"]),
                "action_mask": torch.zeros_like(transition["action_mask"]),
            }
        })


def multi_step_td(
        transition: List[TensorDict], 
        multi_step:int, 
        gamma:float, 
        step_reward_lower_bound:float, 
        hint_reward:float, 
        discard_reward:float
    ):
    ''' Takes an entire trajectory and returns the multi-step transition'''
    # total_reward = 0
    next_idxs = []
    for i, t in enumerate(transition):
        # if step_reward_lower_bound is not None and t["reward"].item() < step_reward_lower_bound:
        #     t["reward"] = torch.zeros_like(t["reward"]) + step_reward_lower_bound
        # total_reward += max(t["reward"].item(), 0)
        if i + multi_step < len(transition):
            # t['bootstrap'] = torch.ones_like(t['reward'])
            next_idxs.append(i + multi_step)
        else:
            # t['bootstrap'] = torch.ones_like(t['reward'])
            next_idxs.append(len(transition) - 1)
        # t['bootstrap'] = torch.zeros_like(t['reward']) + 

        if t["action"]["action"].item() <= 1:
            t["reward"] += discard_reward
        
        if t["action"]["action"].item() >= 4:
            t["reward"] += hint_reward

    for i, t in enumerate(transition):
        t['actual_reward'] = t['reward'].clone()
        # t["bootstrap"] = torch.zeros_like(t["reward"]) + (total_reward / 10)/2 + 0.5
        t['bootstrap'] = torch.ones_like(t['reward'])

        for j in range(i+1, next_idxs[i]+1):
            t['reward'] += gamma ** (j - i) * transition[j]['reward']
        if i != next_idxs[i]:
            t['next'] = TensorDict({
                "observation": transition[next_idxs[i]]['observation'].clone(),
                "action_mask": transition[next_idxs[i]]['action_mask'].clone()
            })
    
    return transition


class BatchRunner():
    def __init__(
            self, 
            env_fn:  Callable[[], HanaEnv],
            sequence_length:int ,
            multi_step:int,
            n_agents:int, # 1 for iql and >1 for vdn
            step_reward_lower_bound:float,
            debug:bool,
            hint_reward:float,
            discard_reward:float,
            num_threads:int = mp.cpu_count() - 1,
        ):
        self.env_fn = env_fn
        self.sequence_length = sequence_length
        self.num_threads = num_threads
        self.multi_step = multi_step
        self.n_agents = n_agents
        self.step_reward_lower_bound = step_reward_lower_bound
        self.debug = debug
        self.hint_reward = hint_reward
        self.discard_reward = discard_reward

    def run_episode(
            self, 
            agent:DQNAgent, 
            epsilon:float, 
            seed:int, 
            env:HanaEnv
        ) -> List[TensorDict]:
        results = []
        state = env.reset(seed=seed)
        for player in env.agent_iter():
            if state['done'].item():
                break
            else:
                action = agent.act(state[player], epsilon)
            new_state = env.step(action['action'], action['greedy_action'])

            results.append(
                TensorDict({
                    "observation": state[player]['observation'],
                    "action_mask": state[player]['action_mask'],
                    "reward": new_state["next"][player]["reward"],
                    "done": new_state["next"][player]["done"],
                    "action": action,
                    "next":{
                        "observation": new_state["next"][player]["observation"],
                        "action_mask": new_state["next"][player]["action_mask"]
                    }
            }))

            state = new_state['next']
        # return torch.stack(results, 0)
        return results

    def run(
            self,
            agent:DQNAgent, 
            buffer:TensorDictPrioritizedReplayBuffer, 
            epsilon:float,
            seed: int,
            min_steps:int,
            return_stats:bool = False
        ):
        env = self.env_fn()
        if return_stats:
            stats = StatsCollector(env.out_dim)

        with torch.no_grad():
            while min_steps > 0:
                new_seed = (seed + min_steps + 1) * 99999999 % 999999997
                trajectory = self.run_episode(agent, epsilon, new_seed, env)
                trajectory = multi_step_td(trajectory, self.multi_step, agent.gamma, self.step_reward_lower_bound, self.hint_reward, self.discard_reward)
                
                priorities = []
                for transition in trajectory:
                    priority = agent.compute_priority(transition)
                    priorities.append(priority)

                if return_stats:
                    stats.store_episode(trajectory, priorities, epsilon)

                if len(trajectory) == 0:
                    continue

                while len(trajectory) < self.sequence_length:
                    trajectory.append(zeros_like_td(trajectory[0]))
                trajectory = torch.stack(trajectory, 0)
                
                idx = buffer.extend(trajectory)
                buffer.update_priority(idx, priorities)

                min_steps -= len(trajectory)
        
        if return_stats:
            return stats
        else:
            return None
        
    def evaluate(self, agent:DQNAgent, num_episodes:int, seed:int, epsilon:float):
        with torch.no_grad():
            env = self.env_fn()
            rewards = []
            steps = []
            max_rewards = []
            actions = [0] * env.out_dim
            for i in trange(num_episodes):
                step = 0
                max_reward = 0
                new_seed = (i * (seed + 1)) * 999999 % 999999997
                state = env.reset(seed=new_seed)
                total_reward = 0
                for player in env.agent_iter():
                    if state['done'].item():
                        break
                    else:
                        action = agent.act(state[player], epsilon=epsilon)
                        actions[action['action']] += 1
                    new_state = env.step(action['action'], action['greedy_action'])
                    total_reward += new_state["next"][player]["reward"].detach().cpu().item()
                    max_reward = max(max_reward, total_reward)
                    state = new_state['next']
                    step += 1

                rewards.append(total_reward)
                max_rewards.append(max_reward)
                steps.append(step)
            return {
                "mean": sum(rewards) / len(rewards),
                "std": torch.std(torch.tensor(rewards)).item(),
                # "rewards": rewards
                "avg_len": sum(steps) / len(steps),
                "max_reward": max(max_rewards),
                "avg_max_reward": sum(max_rewards) / len(max_rewards),
                "max_len": max(steps),
                "min_len": min(steps),
                "actions": actions
            }
        
    def evaluate_multi(self, agent_1:DQNAgent,agent_2:DQNAgent, num_episodes:int, seed:int, epsilon:float):
        with torch.no_grad():
            env = self.env_fn()
            rewards = []
            steps = []
            max_rewards = []
            actions = [0] * env.out_dim
            for i in trange(num_episodes):
                step = 0
                max_reward = 0
                new_seed = (i * (seed + 1)) * 999999 % 999999997
                state = env.reset(seed=new_seed)
                total_reward = 0
                for player in env.agent_iter():
                    if player == "player_0":
                        agent = agent_1
                    else:
                        agent = agent_2
                        
                    if state['done'].item():
                        break
                    else:
                        action = agent.act(state[player], epsilon=epsilon)
                        actions[action['action']] += 1
                    new_state = env.step(action['action'], action['greedy_action'])
                    total_reward += new_state["next"][player]["reward"].detach().cpu().item()
                    max_reward = max(max_reward, total_reward)
                    state = new_state['next']
                    step += 1

                rewards.append(total_reward)
                max_rewards.append(max_reward)
                steps.append(step)
            return {
                "mean": sum(rewards) / len(rewards),
                "std": torch.std(torch.tensor(rewards)).item(),
                # "rewards": rewards
                "avg_len": sum(steps) / len(steps),
                "max_reward": max(max_rewards),
                "avg_max_reward": sum(max_rewards) / len(max_rewards),
                "max_len": max(steps),
                "min_len": min(steps),
                "actions": actions
            }
        
    def run_parallel(
            self,
            agent:DQNAgent, 
            buffer:TensorDictPrioritizedReplayBuffer, 
            epsilon:float,
            seed: int,
            min_steps:int,
        ):
        threads = []
        for i in range(self.num_threads):
            thread = mp.Process(target=self.run, args=(agent, buffer, epsilon, seed + i, min_steps // self.num_threads))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

     