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
        self.device = device
        self.name = 'DQN' + ('-VDN' if vdn else '') + ('-IQL' if not vdn else '') + ('-QMIX' if not vdn else '') + ('-Noisy' if noisy else '') + ('-Distributional' if distributional else '') + ('-Dueling' if dueling else '') + ('-Double' if double else '') + ('-MultiStep' if multi_step > 1 else '')

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
        observation = torch.tensor(state['observation'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(state['action_mask']).to(self.device)
        a = self.policy.act(observation, actions, epsilon)
        actions = a['actions']
        greedy = a['greedy_actions']

        return {
            'action': actions.detach().cpu().numpy(),
            'greedy_action': greedy.detach().cpu().numpy()
        }
    

    def compute_loss_and_priority(self, batch: Dict[str, torch.Tensor]):
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
            log=True,
    ):
        '''
        This function computes the TD error.
        '''
        policy = self.policy(batch['observation'], batch['action_mask'], batch['action']['action'])
        # if log:
        #     print(policy)
        with torch.no_grad():
            target = self.target(batch['next_observation'], batch['next_action_mask'], None)

        terminals = batch['done'].float()
        rewards = batch['reward'].float()
        # if self.vdn:
        #     policy = policy.view(policy.size(0), -1, policy.size(-1))
        #     target = target.view(target.size(0), -1, target.size(-1))
        policy = policy["actual_q"]
        target = target["q"].argmax(-1).float()

        # target = torch.cat(
        #     [
        #         target[self.multi_step - 1:],
        #         torch.zeros_like(target[:self.multi_step - 1])
        #     ],
        #     dim=0
        # )

        # mask = torch.arange(0, policy.size(0), device=policy.device) + 1
        # mask = mask < self.multi_step
        # mask = mask.float()

        # AND the mask with the terminal mask
        mask = (1 - terminals)
        target = rewards + (self.gamma ** self.multi_step)  * target

        error = target.detach() - policy
        error = error * mask    
        # assert False
        return error


