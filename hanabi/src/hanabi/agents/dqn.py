"""
    This file contains the implementation of the DQN agent.
    This model is based on the DQN algorithm, which is a deep reinforcement learning algorithm.
    It supports all the configurations available in the Rainbow DQN paper.
"""
import torch
import torch.nn as nn
from torchrl.modules import MLP, NoisyLinear
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage
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

'''
    This is the implementation of the DQN agent.
'''

class DQNAgent:
    '''
    This class represents the DQN agent.
    '''
    def __init__(
            self, 
            # model parameters
            state_dim, 
            action_dim, 
            hidden_dim=256, 
            depth=2, 
            noisy=False, 
            distributional=False, 
            duel=True, 
            # agent parameters
            double=True, 
            tau=1e-3, 
            gamma=0.99,
            policy_update_freq=4,
            target_update_freq=1000,
            # optimizer parameters
            lr=1e-3, 
            # replay buffer parameters
            capacity=100_000,
            prioritized=False,
            batch_size=256,
            # exploration parameters
            start_epsilon=1.0,
            end_epsilon=0.01,
            epsilon_decay=0.995,
            # additional 
            sad=False,
            shuffle_observation=False,
        ):
        '''
        This function initializes the DQN agent.
        :param state_dim: The dimension of the state.
        :param action_dim: The dimension of the action.
        :param hidden_dim: The dimension of the hidden layer.
        :param hidden_depth: The depth of the hidden layer.
        :param noisy: Whether to use noisy layers.
        :param distributional: Whether to use distributional DQN.
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.noisy = noisy
        self.distributional = distributional
        self.duel = duel
        self.double = double
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = start_epsilon

        self.step_counter = 0
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq

        self.sad = sad
        self.shuffle_observation = shuffle_observation

        # make the buffer
        if prioritized:
            self.buffer = TensorDictPrioritizedReplayBuffer(alpha=0.6, beta=0.4, storage=LazyTensorStorage(capacity), batch_size=batch_size)
        else:
            self.buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(capacity), batch_size=batch_size)

        self.policy = DQN(state_dim, action_dim, hidden_dim, depth, noisy, distributional, duel)
        if double:
            self.target = DQN(state_dim, action_dim, hidden_dim, depth, noisy, distributional, duel)
            self.target.load_state_dict(self.policy.state_dict())
            self.target.eval()
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def act(self, state:TensorDict):
        '''
        This function returns the action to take.
        :param state: The state of the environment.
        :param epsilon: The epsilon value for epsilon-greedy.
        :return: The action to take.
        '''
        if torch.rand(1).item() < self.epsilon:
            mask = state['action_mask'] # bool mask over action space
            return torch.randint(0, mask.size(1), (mask.size(0),)).masked_fill(~mask, -1)
        else:
            with torch.no_grad():
                q = self.model(state)
                return q.argmax(dim=-1)
    
    def step(self, transition:TensorDict, training=True):
        '''
        This function increments the step counter.
        '''
        self.step_counter += 1
        loss = None
        if training:
            # add the transition to the buffer
            self.buffer.add(transition)

            # decay the epsilon value
            self.epsilon = max(self.end_epsilon, self.epsilon * self.epsilon_decay)

            # check if we need to update the policy
            if self.step_counter % self.policy_update_freq == 0:
                loss = self.update()
                
            # check if we need to update the target
            if self.double and self.step_counter % self.target_update_freq == 0:
                self.update_target()
        
        return loss



    def update(self):
        '''
        This function updates the model.
        :param state: The state of the environment.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state of the environment.
        :param done: Whether the episode is done.
        :param optimizer: The optimizer to use.
        :param gamma: The discount factor.
        '''
        # sample a batch of transitions
        batch = self.buffer.sample()

        # unpack the batch
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']
        done = batch['done']

        # calculate the Q value
        state_q = self.policy(state)

        with torch.no_grad():
            if self.double:
                next_state_q = self.target(next_state)
            else:
                next_state_q = self.policy(next_state)
            
            next_state_q = next_state_q.max(dim=-1).values

        # calculate the target Q value
        # Q(s, a) = R + γ * max_a' Q(s', a')
        target_q = reward + self.gamma * next_state_q * (1 - done)

        # calculate the loss
        loss = torch.nn.HuberLoss()
        loss = loss(state_q.gather(1, action.unsqueeze(1)), target_q.unsqueeze(1))

        # update the td_error for prioritized replay buffer
        if isinstance(self.buffer, TensorDictPrioritizedReplayBuffer):
            td_error = (state_q.gather(1, action.unsqueeze(1)) - target_q.unsqueeze(1)).abs()
            batch['td_error'] = td_error
            self.buffer.update_tensordict_priority(batch)

        # update the model
        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient
        # We use 100 as the maximum gradient norm because it is a common value used in the literature.
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 100)

        # update the model
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        '''
        This function does a soft update of the target network.
        To do a hard update, simply do:
        self.target.load_state_dict(self.policy.state_dict())
        or set tau to 1.
        '''
        if self.double:
            for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

