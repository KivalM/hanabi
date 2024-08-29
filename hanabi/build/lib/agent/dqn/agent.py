import torch
import torch.nn as nn
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from .net import DQN

class DQNAgent:
    '''
    This class represents the DQN agent.
    '''
    def __init__(
            self, 
            # policy nets
            policy_net:nn.Module,
            target_net:nn.Module,
            optimizer:torch.optim.Optimizer,
            replay_buffer:TensorDictReplayBuffer,
            # agent parameters
            double=True, 
            tau=1e-3, 
            gamma=0.99,
            policy_update_freq=256,
            target_update_freq=1024,
            # exploration parameters
            start_epsilon=1.0,
            end_epsilon=0.01,
            anneal_steps=1000*100,
            # additional 
            sad=False,
            shuffle_observation=False,
            # checkpoint frequency
            checkpoint_freq=1024*10
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
                # make the buffer
        self.buffer = replay_buffer
        self.policy_net = policy_net
        self.gamma = gamma
        self.optimizer = optimizer
        self.double = double

        if double:
            self.tau = tau
            self.target = target_net
            self.target.load_state_dict(self.policy.state_dict())
            self.target.eval()


        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon = start_epsilon
        self.epsilon_decay = (start_epsilon - end_epsilon) / anneal_steps

        self.step_counter = 0
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq

        self.sad = sad
        self.shuffle_observation = shuffle_observation

        self.done = False

        

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
            self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)

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

