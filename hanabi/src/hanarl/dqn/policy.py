import torch.nn as nn
from torch import Tensor
from torchrl.modules import MLP, NoisyLinear
import torch
from typing import Dict

class DQNPolicy(nn.Module):
    '''
    This class represents the DQN model.
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
    ):
        super(DQNPolicy, self).__init__()
        
        self.dueling = dueling
        self.distributional = distributional
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.noisy = noisy
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        if noisy:
            layer_class = NoisyLinear
        else:
            layer_class = nn.Linear

        # MLP is just a nn.Sequential with layers
        self.net = MLP(
            in_features=in_dim,
            out_features=hidden_dim,
            num_cells=hidden_dim,
            depth=depth,
            layer_class=layer_class,
            activate_last_layer=True,
            activation_class=nn.ReLU
        )

        if distributional:
            out_dim = (n_atoms, out_dim)

        if dueling:
            self.value = nn.Linear(hidden_dim, 1)
            self.advantage = MLP(hidden_dim, out_dim)
        else:
            self.out_layer = MLP(hidden_dim, out_dim)
    
    def act(
            self,
            observation:Tensor,
            legal_actions:Tensor,
            epsilon:float,
    ) -> Dict[str, Tensor]:
        '''
        acts on a single observation
        '''
        assert observation.dim() in [1,2], f"Expected observation to have dimension 2, got {observation.dim()}"
        assert legal_actions.dim() in [1,2], f"Expected legal_actions to have dimension 2, got {legal_actions.dim()}"

        assert observation.dim() == 1, "Only 1 env is supported for now"

        # all the dimensions should be [batch_size, dim]
        state_encoding = self.net(observation)

        if self.dueling:
            q = self._duel(state_encoding, legal_actions)
        else:
            q = self.out_layer(state_encoding)


        # distribute the q values
        if self.distributional:
            q = self._distributional(q)

        assert q.dim() == 1, f"Expected q to have dimension 1, got {q.dim()}"

        # compute legal q values
        legal_q = (1 + q - q.min()) * legal_actions
        # compute the new greedy actions
        greedy_actions = legal_q.argmax(0).detach()
        # choose a random action from each legal action set
        random_actions = legal_actions.float().multinomial(1).squeeze(-1)
        # if the random number is greater than epsilon, choose the greedy action
        greedy_mask = torch.rand(1, device=random_actions.device) > epsilon
        # # choose the actions based on the epsilon greedy policy
        actions = torch.where(greedy_mask, greedy_actions, random_actions).squeeze(0)

        # [batch_size]
        return {
            "actions": actions,
            "greedy_actions": greedy_actions,
        }



    def forward(
            self, 
            # [seq_len, batch_size, obs_dim] or [batch_size, obs_dim]
            observation:Tensor,
            # [seq_len, batch_size, action_dim] or [batch_size, action_dim]
            legal_actions:Tensor,
            # [seq_len, batch_size, action_dim] or [batch_size, action_dim]
            actions:Tensor,
    ) -> Dict[str, Tensor]:
        # check that the dimensions are either 2 or 3 [seq_len, batch_size, obs_dim] or [batch_size, obs_dim]
        assert observation.dim() in [2, 3], f"Expected observation to have dimension 2 or 3, got {observation.dim()}"

        two_dim = observation.dim() == 2
        # TODO: temporary fix, need to handle the case where the observation is 3D
        assert two_dim, "Only 1 step is supported for now"

        if two_dim:
            observation = observation.unsqueeze(0).float()
            legal_actions = legal_actions.unsqueeze(0)
            actions = actions.unsqueeze(0) if actions is not None else None

        # all the dimensions should be [seq_len(/1), batch_size, dim]
        state_encoding = self.net(observation)

        if self.dueling:
            q = self._duel(state_encoding, legal_actions)
        else:
            q = self.out_layer(state_encoding)


        if self.distributional:
            q = self._distributional(q)


        assert q.dim() == 3, f"Expected q to have dimension 3, got {q.dim()}"
        
        # print("Q values")
        # print(q)
        # compute legal q values
        legal_q = (1 + q - q.min()) * legal_actions
        # print("Legal Q values")
        # print(legal_q)
        # compute the q values of the actual actions taken
        if actions is not None:
            actual_q = legal_q.gather(2, actions.unsqueeze(2)).squeeze(2)
        else:
            actual_q = None

        # compute the new greedy actions
        greedy_actions = legal_q.argmax(2).detach()

        if two_dim:
            greedy_actions = greedy_actions.squeeze(0)
            legal_q = legal_q.squeeze(0)
            if actual_q is not None:
                actual_q = actual_q.squeeze(0)
            state_encoding = state_encoding.squeeze(0)

        return {
            "q": legal_q,
            "greedy_actions": greedy_actions,
            "actual_q": actual_q,
            "state_encoding": state_encoding,
        }


    def _duel(self, state_encoding, legal_actions):
        # calculate the value and advantage
        # Value: V(s) = E[R|s]
        # Advantage: A(s, a) = Q(s, a) - V(s)
        # Q(s, a) = V(s) + A(s, a)
        # This is a way to reduce the variance of the Q values  by subtracting the value function from the Q function
        # We use the legal actions to mask the advantage in order to avoid the computation of the advantage for illegal actions
        value = self.value(state_encoding)
        advantage = self.advantage(state_encoding)
        q = value + (advantage - advantage.mean(-1, keepdim=True)) * legal_actions
        return q

    def _distributional(self, q):
        two_dim = q.dim() == 2
        four_dim = q.dim() == 4
        if two_dim:
            q = q.unsqueeze(0)
        if four_dim:
            q = q.squeeze(0)

        # reshape the output tensor to (batch_size, 4, -1)
        q = q.view(q.size(0), self.n_atoms, -1)
        
        # compute the probabilities
        q = nn.functional.softmax(q, dim=-1)

        # compute the atoms
        atoms = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=q.device).unsqueeze(0).unsqueeze(2)
        # compute the q values
        q = (atoms * q).sum(1)

        if two_dim:
            q = q.squeeze(0)

        if four_dim:
            q = q.unsqueeze(0)
        
        return q
