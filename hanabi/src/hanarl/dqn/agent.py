import torch
import torch.nn as nn
from typing import Dict, Union
from .policy import DQNPolicy
from tensordict import TensorDict

class DQNAgent(nn.Module):
    '''
    This class represents the DQN Agent
    '''
    def __init__(
            self,
            # standard arguments
            in_dim: int,
            out_dim: int,
            hidden_dim: int,
            depth: int,
            # noisy arguments
            noisy: bool,
            noise_std: float,
            # distributional arguments
            distributional: bool, # whether to use distributional DQN
            n_atoms: int, # number of atoms in the distribution
            v_min: float, # minimum value of the distribution
            v_max: float, # maximum value of the distribution
            # dueling arguments
            dueling: bool,
            # vdn, iql, qmix arguments
            vdn: bool,
            # multi-step arguments for n-step q-learning 0 for standard q-learning
            multi_step: int,
            gamma: float, # discount factor for multi-step q-learning
            tau: float, # target network update rate - 1 for hard update
            # double q-learning arguments
            double: bool,
            device: str, # device to run the model on

    ):
        super(DQNAgent, self).__init__()
        self.policy = DQNPolicy(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            noisy=noisy,
            noise_std=noise_std,
            distributional=distributional,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            dueling=dueling
        ).to(device)

        self.target = DQNPolicy(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            noisy=noisy,
            noise_std=noise_std,
            distributional=distributional,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            dueling=dueling
        ).to(device)

        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.tau = tau
        self.double = double
        self.device = device
        self.name = 'DQN' + ('-VDN' if vdn else '') + ('-IQL' if not vdn else '')  + ('-Noisy' if noisy else '') + ('-Distributional' if distributional else '') + ('-Dueling' if dueling else '') + ('-Double' if double else '') + ('-MultiStep' if multi_step > 1 else '')

    def update_target(self):
        '''
        This function updates the target network.
        '''
        for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

    def act(self, state: Dict[str, Union[torch.Tensor, int]], epsilon: float):
        '''
        This function returns the action to take.
        :param state: The state of the environment.
        :param epsilon: The epsilon value for epsilon-greedy.
        :return: The action to take.
        '''
        observation = state['observation'].float().to(self.device)
        actions = state['action_mask'].to(self.device)
        observation = observation.squeeze(0)
        actions = actions.squeeze(0)
        a = self.policy.act(observation, actions, epsilon)
        actions = a['actions'].detach().cpu().item()
        greedy = a['greedy_actions'].detach().cpu().item()

        return TensorDict({
            'action': [actions],
            'greedy_action': [greedy]
        }, batch_size=1)
    
    def compute_priority(self, transition: TensorDict):
        '''
        This function computes the priority for a single transition.
        This gets called during the sampling process.
        '''

        if self.vdn:
            print(transition['observation'].size())
            num_agents, obs_size = transition['observation'].size()
        else:
            num_agents, obs_size = transition['observation'].size()
        
        observation = transition['observation'].float().to(self.device)
        legal_actions = transition['action_mask'].to(self.device)
        action = transition['action']['action'].to(self.device)

        next_observation = transition['next', 'observation'].float().to(self.device)
        next_legal_actions = transition['next', 'action_mask'].to(self.device)

        reward = transition['reward'].float().to(self.device)   
        bootstrap = transition['bootstrap'].float().to(self.device)

        policy = self.policy(observation, legal_actions, action)
        next_policy = self.policy(next_observation, next_legal_actions, None)
        target = self.target(next_observation, next_legal_actions, next_policy['greedy_actions'])
        assert reward.size() == bootstrap.size(), "Reward: {} Bootstrap: {}".format(reward.size(), bootstrap.size())
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target['actual_q']
        priority = torch.abs(target - policy['actual_q']).detach().cpu()
        assert priority.size() == (1, 1), "Priority: {}".format(priority.size())
        return priority


    def compute_loss_and_priority(self, batch: Dict[str, torch.Tensor]):
        '''
        This function computes the loss.
        '''
        error = self.td_error(batch)

        loss = nn.SmoothL1Loss(reduction="none")(error, torch.zeros_like(error))
        loss = loss.mean()

        priority = torch.abs(error).detach().cpu().numpy()

        return {
            'loss': loss,
            'priority': priority
        }
    
    def td_error(
            self,
            batch: TensorDict,
            log=True,
    ):
        '''
        This function computes the TD error.

        :param batch: The batch of transitions with dims 
            [batch_size, num_agents, obs_size]
        '''
        assert batch.dim() == 1, "Batch: {}".format(batch.size())
        if self.vdn:
            # expand batch to [batch_size, obs_size]
            batch = batch.view(-1, batch.size(-1))
        else:
            # shrink batch to [batch_size, obs_size] from [batch_size, num_agents, obs_size]
            observation = batch['observation'].float().to(self.device).squeeze(1)
            # print(batch.size())
            # print(observation.size())
        
            action_mask = batch['action_mask'].to(self.device).squeeze(1)
            # print(action_mask.size())
            action = batch['action']['action'].to(self.device).squeeze(1)
            # print(action.size())
            next_observation = batch['next', 'observation'].float().to(self.device).squeeze(1)
            next_action_mask = batch['next', 'action_mask'].to(self.device).squeeze(1)
            # print(next_observation.size())
            # print(next_action_mask.size())
            assert next_action_mask.size() == action_mask.size()
            assert action.size() == (batch.size()), "Action: {} Batch: {}".format(action.size(), batch.size())
            assert next_observation.size() == observation.size() == (batch.size(0), self.policy.in_dim), "Observation: {} Batch: {}".format(observation.size(), (batch.size(), self.policy.in_dim))
            terminals = batch['done'].float().squeeze(1).squeeze(1)
            rewards = batch['reward'].float().squeeze(1).squeeze(1)
            bootstrap = batch['bootstrap'].float().squeeze(1).squeeze(1)

        policy = self.policy(observation, action_mask, action)
        # print(policy)
        with torch.no_grad():
            target = self.target(next_observation, next_action_mask, None)
            # print(target)

 

        policy = policy["actual_q"]
        # print("Policy: {}".format(policy.size()))
        # print(policy)
        target = target["greedy_q"]
        # print("Target: {}".format(target.size()))
        # print(target)
        # print("Rewards: {}".format(rewards.size()))
        # print(rewards)
        # print("Bootstrap: {}".format(bootstrap.size()))
        # print(bootstrap)
        non_final_mask = 1 - terminals
        target = rewards + bootstrap * (self.gamma ** self.multi_step) * target * non_final_mask
        # print("Target: {}".format(target.size()))
        # print(target)
        error = target.detach() - policy
        error = error 
        # print("Error: {}".format(error.size()))
        # print(error)
        # assert False
        return error
    
    def save(self, path):
        '''
        This function saves the model.
        '''
        torch.save(self, path)
    
    def load(path):
        '''
        This function loads the model.
        '''
        return torch.load(path)
    
    def copy(self):
        '''
        This function copies the model.
        '''
        agent =  DQNAgent(
            in_dim=self.policy.in_dim,
            out_dim=self.policy.out_dim,
            hidden_dim=self.policy.hidden_dim,
            depth=self.policy.depth,
            noisy=self.policy.noisy,
            distributional=self.policy.distributional,
            n_atoms=self.policy.n_atoms,
            v_min=self.policy.v_min,
            v_max=self.policy.v_max,
            dueling=self.policy.dueling,
            vdn=self.vdn,
            multi_step=self.multi_step,
            gamma=self.gamma,
            tau=self.tau,
            double=self.double,
            device=self.device
        )
        agent.policy.load_state_dict(self.policy.state_dict())
        agent.target.load_state_dict(self.target.state_dict())
        return agent


