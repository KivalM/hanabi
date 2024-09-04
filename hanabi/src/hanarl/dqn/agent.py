import torch
import torch.nn as nn
from typing import Dict, Union
from .policy import DQNPolicy

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
            device: str, # device to run the model on
            # double q-learning arguments
            double: bool,
    ):
        super(DQNAgent, self).__init__()
        self.policy = DQNPolicy(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            noisy=noisy,
            distributional=distributional,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            dueling=dueling
        )        

        self.target = DQNPolicy(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            noisy=noisy,
            distributional=distributional,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            dueling=dueling
        )

        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.tau = tau
        self.device = device

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
        a = self.policy.act(state, epsilon)
        actions = a['actions']
        greedy = a['greedy_actions']

        return {
            'actions': actions.detach().cpu().numpy(),
            'greedy_actions': greedy.detach().cpu().numpy()
        }
    
    def compute_priority(self, state: Dict[str, Union[torch.Tensor, int]]):
        '''
        This function computes the priority of a transition.
        '''
        return self.policy.compute_priority(state)
    

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        '''
        This function computes the loss.
        '''
        error = self.td_error(batch)

        loss = nn.SmoothL1Loss()(error, torch.zeros_like(error))
        loss = loss.mean()

        priority = torch.abs(error).detach().cpu().numpy()

        return {
            'loss': loss,
            'priority': priority
        }
    
    def td_error(
            self,
            batch: Dict[str, torch.Tensor],
    ):
        '''
        This function computes the TD error.
        '''
        policy = self.policy(batch['observation'], batch['legal_actions'], batch['action'])

        with torch.no_grad():
            target = self.target(batch['next_observation'], batch['next_legal_actions'])

        terminals = batch['terminal'].float()
        rewards = batch['reward'].float()

        if self.vdn:
            policy = policy.view(policy.size(0), -1, policy.size(-1))
            target = target.view(target.size(0), -1, target.size(-1))

        target = torch.cat(
            [
                target[self.multi_step - 1:],
                torch.zeros_like(target[:self.multi_step - 1])
            ],
            dim=0
        )

        mask = torch.arange(0, policy.size(0), device=policy.device) + 1
        mask = mask < self.multi_step
        mask = mask.float()

        # AND the mask with the terminal mask
        mask = mask * (1 - terminals)

        target = rewards + (self.gamma ** self.multi_step)  * target

        error = policy - target.detach()
        error = error * mask

        return error


