from ..agent.base import BaseAgent
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

def run_episode_single_agent(
        env:OrderEnforcingWrapper,
        agent: BaseAgent,
    ):
    """Runs a single episode of the game with the given agent.

    Args:
        agents: A dictionary of agents to run the episode with. The key should match the player id.
        env: The environment to run the episode in.

    Returns:
        episode_returns: A named tuple containing the returns and the number of steps in the episode.
    """
    env.reset()
    
    # store last observation for each agent
    last_observations = {
        a: None
        for a in env.agents
    }

    last_actions = {
        a: None
        for a in env.agents
    }

    rewards = {
        a: 0.0
        for a in env.agents
    }

    max_rewards = rewards.copy()
    # iterate through the environment until the episode is over
    for player in env.agent_iter():
        observation, reward, terminated, truncated, _ = env.last()
  

        done = terminated or truncated

        if done:
            action = None
        else:
            action = agent.act(observation)

        rewards[player] += reward
        max_rewards[player] = max(rewards[player], max_rewards[player])

        # save the results to the replay buffer
        # TODO: Episodic memory
        # agents[player].store(last_observations[player], last_actions[player], observation, reward, done)
        agent.step((last_observations[player], last_actions[player],  reward, observation,done))
        
        # store the last observation and action
        last_actions[player] = action
        last_observations[player] = observation
        # perform the action
        env.step(action)


    agent.save_episode_results(max(max_rewards.values()), max(rewards.values()))











# def run_episode_single_agent(
#         env:TransformedEnv,
#         agent:BaseAgent,
#     ) :
#     """Runs a single episode of the game with the given agent.

#     Args:
#         agents: A dictionary of agents to run the episode with. The key should match the player id.
#         env: The environment to run the episode in.

#     Returns:
#         episode_returns: A named tuple containing the returns and the number of steps in the episode.
#     """
#     rewards = {
#         agent: 0.0
#         for agent in env.agents
#     }

#     max_rewards = rewards.copy()
#     # iterate through the environment until the episode is over
#     state = env.reset()
#     for player in env.agent_iter():

#         state = state[player]

#         if state['done']:
#             action = None
#             break
#         action = agent.act(state)
#         with torch.no_grad():
#             env_action = TensorDict({player: {"action": [action]}})

#         next = env.step(env_action)['next']
#         rewards[player] += next[player]['reward']
#         max_rewards[player] = max(max_rewards[player], rewards[player])
#         with torch.no_grad():
#             transition = TensorDict({
#                 "state": {
#                     'observation': state['observation', 'observation'],
#                     'action_mask': state['action_mask'],
#                     'greedy_action': None,
#                 },
#                 "action": action,
#                 "reward": next[player]['reward'].item(),
#                 "next_state":{
#                     'observation': next[player, 'observation', 'observation'],
#                     'action_mask': next[player, 'action_mask'],
#                     'greedy_action': None,
#                 },
#                 "done": next['done'],
#             })

#         agent.step(transition)

#         state = next

#     agent.save_episode_results(max(max_rewards.values()), max(rewards.values()))
