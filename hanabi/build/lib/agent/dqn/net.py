import torch.nn as nn
from torchrl.modules import MLP, NoisyLinear
from tensordict import TensorDict

class DQN(nn.Module):
    '''
    This class represents the DQN model.
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256, depth=2, noisy=False, distributional=False, duel=False):
        '''
        This function initializes the DQN model.
        :param state_dim: The dimension of the state.
        :param action_dim: The dimension of the action.
        :param hidden_dim: The dimension of the hidden layer.
        :param hidden_depth: The depth of the hidden layer.
        :param noisy: Whether to use noisy layers.
        :param distributional: Whether to use distributional DQN.
        '''
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.noisy = noisy
        self.distributional = distributional
        self.duel = duel

        if noisy:
            layer_class=NoisyLinear
        else:
            layer_class=nn.Linear

        self.net = MLP(in_features=state_dim, out_features=hidden_dim, num_cells=hidden_dim, depth=depth, layer_class=layer_class, activate_last_layer=True)

        if duel:
            self.value = MLP(in_features=state_dim, out_features=1, num_cells=hidden_dim, depth=1,)
            self.advantage = MLP(in_features=state_dim, out_features=action_dim, num_cells=hidden_dim, depth=1,)

        if distributional:
            action_dim = (4, action_dim)

        self.out_layer = MLP(in_features=hidden_dim, out_features=action_dim, num_cells=hidden_dim, depth=1, layer_class=layer_class, activate_last_layer=False)

    def forward(self, state:TensorDict, sad=False):
        '''
        This function performs a forward pass.
        :param x: The input tensor.
        :return: The output tensor.
        '''
        observation = state['observation']
        action_mask = state['action_mask']
        
        state_encoding = self.net(observation)

        if self.duel:
            # calculate the value and advantage
            # Value: V(s) = E[R|s]
            # Advantage: A(s, a) = Q(s, a) - V(s)
            # Q(s, a) = V(s) + A(s, a)
            # This is a way to reduce the variance of the Q values  by subtracting the value function from the Q function
            value = self.value(state_encoding)
            advantage = self.advantage(state_encoding)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q = self.out_layer(state_encoding)

        if self.distributional:
            # reshape the output tensor to (batch_size, 4, -1)
            q = q.view(q.size(0), 4, -1)
        else:
            # reshape the output tensor to (batch_size, action_dim)
            q = q.squeeze(1)

        # mask invalid actions
        q = q + action_mask

        return q