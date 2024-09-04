from ..dqn.agent import DQNAgent
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

def run_episode_single_agent_vdn(
        agent: DQNAgent,
        eps: float,
        env: OrderEnforcingWrapper,
        seed: int
    ):
    """Runs a single episode of the game with the given agent."""
    env.reset(seed=seed)
    ep_transitions = []
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
        env._accumulate_rewards()

        done = terminated or truncated

        if done:
            action = None
        else:
            action = agent.act(observation, eps)

        rewards[player] += reward
        max_rewards[player] = max(rewards[player], max_rewards[player])

        # save the results to the replay buffer
        # TODO: Episodic memory
        # agents[player].store(last_observations[player], last_actions[player], observation, reward, done)
        # agent.step((last_observations[player], last_actions[player],  reward, observation,done))
        ep_transitions.append((last_observations[player], last_actions[player],  reward, observation,done))
        # store the last observation and action
        last_actions[player] = action
        last_observations[player] = observation
        # perform the action
        env.step(action)

    return {
        "transitions": ep_transitions,
        "max_rewards": max_rewards,
        "rewards": rewards
    }


def collect_data(env, agent, eps, seed, steps=1000, multi_step=1):
    data = []
    while len(data) < steps:
        # generate a random seed
        new_seed = (seed + len(data) + 1) * 997 % 999999999
        ep_data = run_episode_single_agent_vdn(agent, eps, env, new_seed)
        data.extend(ep_data["transitions"])
        # # count multi-step rewards
        # for i, transition in enumerate(ep_data["transitions"]):
        #     if i + multi_step < len(ep_data["transitions"]):
        #         reward = sum([t[2] for t in ep_data["transitions"][i:i+multi_step]])

        #         data.append((transition[0], transition[1], reward, transition[3], transition[4]))
        #     else:
        #         reward = sum([t[2] for t in ep_data["transitions"][i:]])
        #         reward += sum([t[2] for t in ep_data["transitions"][:multi_step - len(ep_data["transitions"]) + i]])
    
    return data