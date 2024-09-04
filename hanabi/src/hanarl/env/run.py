from ..dqn.agent import DQNAgent
from tensordict import TensorDict
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
            action = {
                "action": None,
                "greedy_action": None
            }
        else:
            action = agent.act(observation, eps)

        rewards[player] += reward
        max_rewards[player] = max(rewards[player], max_rewards[player])

        # save the results to the replay buffer
        # TODO: Episodic memory
        # agents[player].store(last_observations[player], last_actions[player], observation, reward, done)
        # agent.step((last_observations[player], last_actions[player],  reward, observation,done))
        if last_actions[player] is not None:
            # ep_transitions.append((last_observations[player], last_actions[player],  reward, observation,done))
            ep_transitions.append(TensorDict({
                "observation": last_observations[player]['observation'],
                "action_mask": last_observations[player]['action_mask'],
                "action": last_actions[player],
                "reward": reward,
                "next_observation": observation['observation'],
                "next_action_mask": observation['action_mask'],
                "done": done
            }))
        # store the last observation and action
        last_actions[player] = action
        last_observations[player] = observation
        # perform the action
        env.step(action['action'], action['greedy_action'])

    return {
        "transitions": ep_transitions,
        "max_rewards": max_rewards,
        "rewards": rewards
    }


def collect_data(env, agent, buffer, eps, seed, steps=1000, multi_step=1):
        step = 0
        while step < steps:
            # generate a random seed
            new_seed = (seed + step + 1) * 997 % 999999999
            ep_data = run_episode_single_agent_vdn(agent, eps, env, new_seed)
            # data.extend(ep_data["transitions"])
            for i, transition in enumerate(ep_data["transitions"]):
                buffer.add(transition)
                step += 1


def evaluate(env, agent, eps, seed,  num_eps=100):
    rewards = []
    for ep in range(num_eps):
        new_seed = (seed + ep + 1) * 997 % 999999999 
        data = run_episode_single_agent_vdn(agent, eps, env, new_seed)
        rewards.append(data["rewards"])
    
    return {
        # "rewards": rewards,
        "avg_reward": sum([max(r.values()) for r in rewards]) / num_eps,
        "max_reward": max([max(r.values()) for r in rewards])
    }