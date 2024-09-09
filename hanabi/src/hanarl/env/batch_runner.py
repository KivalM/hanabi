from .environment import HanaEnv
from ..dqn.agent import DQNAgent
from typing import Callable
import  torch.multiprocessing as mp
from tensordict import TensorDict
from typing import List
import torch
import torch.multiprocessing as mp
from torchrl.data.replay_buffers import TensorDictPrioritizedReplayBuffer

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


def multi_step_td(transition: List[TensorDict], multi_step:int, gamma:float):
    ''' Takes an entire trajectory and returns the multi-step transition'''
    
    next_idxs = []
    for i, t in enumerate(transition):
        # if t["reward"].item() < 0:
        #     t["reward"] = 0
        
        if i + multi_step < len(transition):
            t['bootstrap'] = 1
            # t['next_idx'] = i + multi_step
            next_idxs.append(i + multi_step)
        else:
            t['bootstrap'] = 0
            # t['next_idx'] = len(transition) - 1
            next_idxs.append(len(transition) - 1)
        # if its not a play action
        if t["action"]["action"].item() >= 4 and t["action"]["action"].item() <= 1:
            t["reward"] += 0.5

    for i, t in enumerate(transition):
        # print action r before and r after 
        for j in range(i+1, next_idxs[i]+1):
            t['reward'] += gamma ** (j - i) * transition[j]['reward']

        t['next'] = TensorDict({
            "observation": transition[next_idxs[i]]['observation'].clone(),
            "action_mask": transition[next_idxs[i]]['action_mask'].clone()
        })
    
    return transition

def other_player(player):
    if player == "player_0":
        return "player_1"
    else:
        return "player_0"

class BatchRunner():
    def __init__(
            self, 
            env_fn:  Callable[[], HanaEnv],
            sequence_length:int ,
            multi_step:int,
            n_agents:int, # 1 for iql and 2 for vdn
            num_threads:int = mp.cpu_count() - 1,
        ):
        self.env_fn = env_fn
        self.sequence_length = sequence_length
        self.num_threads = num_threads
        self.multi_step = multi_step
        self.n_agents = n_agents

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
                    # "next":{
                    #     "observation": new_state["next"][player]["observation"],
                    #     "action_mask": new_state["next"][player]["action_mask"]
                    # }
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
        ):
        env = self.env_fn()
        with torch.no_grad():
            while min_steps > 0:
                new_seed = (seed + min_steps + 1) * 99999999 % 999999997
                trajectory = self.run_episode(agent, epsilon, new_seed, env)
                trajectory = multi_step_td(trajectory, self.multi_step, agent.gamma)
                
                priorities = []
                for transition in trajectory:
                    priority = agent.compute_priority(transition)
                    priorities.append(priority)
                    # print(priority)

                if len(trajectory) == 0:
                    continue

                while len(trajectory) < self.sequence_length:
                    trajectory.append(zeros_like_td(trajectory[0]))
                trajectory = torch.stack(trajectory, 0)
                
                idx = buffer.extend(trajectory)
                buffer.update_priority(idx, priorities)

                min_steps -= len(trajectory)
        
    def evaluate(self, agent:DQNAgent, num_episodes:int, seed:int, epsilon:float):
        with torch.no_grad():
            env = self.env_fn()
            rewards = []
            steps = []
            max_rewards = []
            actions = [0] * env.out_dim
            for i in range(num_episodes):
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
                    # print(new_state["next"]["player_0"]["reward"])
                    # print(new_state["next"]["player_1"]["reward"])
                    max_reward = max(max_reward, total_reward)
                    state = new_state['next']
                    step += 1

                rewards.append(total_reward)
                max_rewards.append(max_reward)
                steps.append(step)
            return {
                "mean": sum(rewards) / len(rewards),
                "std": torch.std(torch.tensor(rewards)),
                # "rewards": rewards
                "avg_len": sum(steps) / len(steps),
                "max_reward": max(max_rewards),
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

     