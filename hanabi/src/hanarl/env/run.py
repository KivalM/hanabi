# from ..dqn.agent import DQNAgent
# from tensordict import TensorDict
# from pettingzoo.utils.wrappers import OrderEnforcingWrapper
# from torchrl.envs import PettingZooEnv
# import torch
# from torchrl.data.replay_buffers import PrioritizedReplayBuffer


# def run_episode_single_agent_vdn(
#         agent: DQNAgent,
#         eps: float,
#         env: PettingZooEnv,
#         seed: int
#     ):
#     """Runs a single episode of the game with the given agent."""
#     state = env.reset(seed=seed)

#     # iterate through the environment until the episode is over
#     for player in env.agent_iter():
#         print(state)
#         if state['done'].item():
#             print('done')
#             break
#         else:
#             action = agent.act(state[player], eps)
#         new_state = env.step(action['action'], action['greedy_action'])
#         print(new_state["next"][player]["reward"])
#         assert state[player] != new_state["next"][player]
#         state= new_state['next']

#     return {
#         # "transitions": ep_transitions,
#         # "max_rewards": max_rewards,
#         # "rewards": rewards,
#     }



# def run_episode_single_agent_vdn_(
#         agent: DQNAgent,
#         eps: float,
#         env: OrderEnforcingWrapper,
#         seed: int
#     ):
#     """Runs a single episode of the game with the given agent."""
#     env.reset(seed=seed)
#     ep_transitions = []
#     # store last observation for each agent
#     last_observations = {
#         a: None
#         for a in env.agents
#     }

#     last_actions = {
#         a: None
#         for a in env.agents
#     }

#     rewards = {
#         a: 0.0
#         for a in env.agents
#     }


#     max_rewards = rewards.copy()
#     # iterate through the environment until the episode is over
#     for player in env.agent_iter():
#         observation, reward, terminated, truncated, _ = env.last()
#         env._accumulate_rewards()

#         done = terminated or truncated

#         if done:
#             action = {
#                 "action": None,
#                 "greedy_action": None
#             }
#         else:
#             action = agent.act(observation, eps)

#         rewards[player] += reward
#         max_rewards[player] = max(rewards[player], max_rewards[player])

#         # save the results to the replay buffer
#         # TODO: Episodic memory
#         # agents[player].store(last_observations[player], last_actions[player], observation, reward, done)
#         # agent.step((last_observations[player], last_actions[player],  reward, observation,done))
#         if last_actions[player] is not None:
#             # ep_transitions.append((last_observations[player], last_actions[player],  reward, observation,done))
#             ep_transitions.append(TensorDict({
#                 "observation": last_observations[player]['observation'],
#                 "action_mask": last_observations[player]['action_mask'],
#                 "action": last_actions[player],
#                 "reward": max(0, reward),
#                 "next_observation": observation['observation'],
#                 "next_action_mask": observation['action_mask'],
#                 "done": done
#             }))
#         # store the last observation and action
#         last_actions[player] = action
#         last_observations[player] = observation
#         # perform the action
#         env.step(action['action'], action['greedy_action'])
        
#     return {
#         "transitions": ep_transitions,
#         "max_rewards": max_rewards,
#         "rewards": rewards,
#     }


# def collect_data(env, agent:DQNAgent, buffer:PrioritizedReplayBuffer, eps, seed, steps=1000, multi_step=1):
#         with torch.no_grad():
#             agent.policy.eval()
#             while steps > 0:
#                 # generate a random seed
#                 new_seed = (seed + steps + 1) * 997 % 999999999
#                 ep_data = run_episode_single_agent_vdn(agent, eps, env, new_seed)

#                 for i, transition in enumerate(ep_data["transitions"]):
#                     # print("Action:", transition["action"]['action'], "Reward:", transition["reward"], 'greedy_action:', transition["action"]['greedy_action'], "Done:", transition["done"], "eps:", eps)
#                     i = buffer.add(transition)
#                     transition_priority = agent.td_error(transition.to(agent.device).unsqueeze(0), False)
#                     # print('Reward:', transition["reward"], "Priority:", torch.abs(transition_priority).detach().item())
#                     buffer.update_priority(i, torch.abs(transition_priority).item())
#                     steps -= 1
            
#             agent.policy.train()



# def evaluate(env, agent, eps, seed,  num_eps=100):
#     with torch.no_grad():
#         agent.policy.eval()
#         rewards = []
#         lengths = []
#         for ep in range(num_eps):
#             new_seed = (seed + ep + 1) * 997 % 999999999 
#             data = run_episode_single_agent_vdn(agent, eps, env, new_seed)
#             rewards.append(data["rewards"])
#             lengths.append(len(data['transitions']))


#         agent.policy.train()


#         return {
#             # "rewards": rewards,
#             "avg_reward": sum([max(r.values()) for r in rewards]) / num_eps,
#             "max_reward": max([max(r.values()) for r in rewards]),
#             "min_reward": min([min(r.values()) for r in rewards]),
#             "avg_length": sum(lengths) / num_eps,
#         }