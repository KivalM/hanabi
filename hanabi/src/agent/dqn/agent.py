import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer
from tensordict import TensorDict
from .net import DQN
import wandb
import numpy as np

class DQNAgent():
    '''
    This class represents the DQN agent.
    '''
    def __init__(
            self, 
            # policy nets
            policy_net:nn.Module,
            target_net:nn.Module,
            optimizer:torch.optim.Optimizer,
            replay_buffer:ReplayBuffer,
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
            checkpoint_freq=1024*10,
            checkpoint_dir='checkpoints',
            # logging
            log=False,
            log_interval=1000,
            # device
            device='cuda'
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
        self.policy_net.to(device)
        self.gamma = gamma
        self.optimizer = optimizer
        self.double = double

        if double:
            self.tau = tau
            self.target_net = target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.to(device)
            self.target_net.eval()


        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon = start_epsilon
        self.epsilon_decay = (start_epsilon - end_epsilon) / anneal_steps
        self.n_times = anneal_steps

        self.step_counter = 0
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq

        self.sad = sad
        self.shuffle_observation = shuffle_observation
        self.device = device
        self.done = False

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir

        self.log = log
        self.log_interval = log_interval

        
        self.max_scores = []
        self.total_scores = []

        

    def act(self, state:TensorDict):
        '''
        This function returns the action to take.
        :param state: The state of the environment.
        :param epsilon: The epsilon value for epsilon-greedy.
        :return: The action to take.
        '''
        if np.random.rand() < self.epsilon:
            mask = state['action_mask']

            # calulate the probability of choosing each legal action randomly
            space = np.arange(len(mask))
            mask = mask / mask.sum()
            action = np.random.choice(space, p=mask)

            return action
        else:
            with torch.no_grad():
                state = {
                    'observation': torch.tensor(state['observation'], device=self.device, dtype=torch.float32),
                    'action_mask': torch.tensor(state['action_mask'], device=self.device, dtype=torch.bool),
                }

                q = self.policy_net(state)
                act =  q.argmax()
                return act.item()
    
    def _store_transition(self, state, action, reward, next_state, done):
        '''
        This function stores the transition in the buffer.
        '''
        if state is not None:
            state_tensor = {
                'observation': torch.tensor(state['observation'], device=self.device, dtype=torch.float32),
                'action_mask': torch.tensor(state['action_mask'], device=self.device, dtype=torch.bool),
            }
            action_tensor = torch.tensor(action, device=self.device, dtype=torch.long)
            reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float32)
            next_state_tensor = {
                'observation': torch.tensor(next_state['observation'], device=self.device, dtype=torch.float32),
                'action_mask': torch.tensor(next_state['action_mask'], device=self.device, dtype=torch.bool),
            }
            done_tensor = torch.tensor(done, device=self.device, dtype=torch.bool)

            self.buffer.add({
                'state': state_tensor,
                'action': action_tensor,
                'reward': reward_tensor,
                'next_state': next_state_tensor,
                'done': done_tensor,
            })
    
    def step(self, transition:any, training=True):
        '''
        This function increments the step counter.
        '''
        self.step_counter += 1
        returns = {}
        if training:
            # add the transition to the buffer
            state, action, reward, next_state, done = transition
            self._store_transition(state, action, reward, next_state, done)

            # decay the epsilon value
            self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)

            # check if we need to update the policy
            if self.step_counter % self.policy_update_freq == 0:
                loss = self.update()
                returns['loss'] = loss
                
            # check if we need to update the target
            if self.double and self.step_counter % self.target_update_freq == 0:
                self.update_target()
            
            # check if we need to save the model
            if self.step_counter % self.checkpoint_freq == 0:
                # pass
                self.save(f'{self.checkpoint_dir}/checkpoint_{self.step_counter}.pt')
                wandb.save(f'{self.checkpoint_dir}/checkpoint_{self.step_counter}.pt')
            
            # check if we need to log
            if self.log and self.step_counter % self.log_interval == 0:
                print(f'Step: {self.step_counter}, Epsilon: {self.epsilon}, Loss: {loss}')

                # avg returns
                avg_max_score = sum(self.max_scores) / len(self.max_scores)
                avg_total_score = sum(self.total_scores) / len(self.total_scores)

                # clear the scores
                self.max_scores = []
                self.total_scores = []

                returns['avg_max_score'] = avg_max_score
                returns['avg_total_score'] = avg_total_score
                returns['epsilon'] = self.epsilon
                returns['step'] = self.step_counter

                # rb size
                returns['replay_buffer_size'] = len(self.buffer)

                wandb.log(returns)
            
            # check if we need to stop
            if self.step_counter >= self.n_times:
                self.done = True



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

        if len(self.buffer) < self.buffer._batch_size:
            return 0



        # sample a batch of transitions
        batch, info = self.buffer.sample(None, True)


        # unpack the batch
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']
        done = batch['done']

        # calculate the Q value
        state_q = self.policy_net(state)

        with torch.no_grad():
            if self.double:
                next_state_q = self.target_net(next_state)
            else:
                next_state_q = self.policy_net(next_state)
            

        # calculate the target Q value
        # Q(s, a) = R + γ * max_a' Q(s', a')
        # done is a boolean tensor
        # convert done to binary tensor
        done = done.float().unsqueeze(1)

        target_q = reward.unsqueeze(1) + (self.gamma * next_state_q * (1 - done))
        target_q = target_q.max(dim=-1).values

        # calculate the loss
        loss = torch.nn.SmoothL1Loss()
        loss = loss(state_q.gather(1, action.unsqueeze(1)), target_q)

        # update the td_error for prioritized replay buffer
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            # calculate the td_error for prioritized replay buffer and add some noise to prevent zero td_error
            td_error = (state_q.gather(1, action.unsqueeze(1)) - target_q.unsqueeze(1)).abs() + 1e-6
            self.buffer.update_priority(info["index"], td_error)

        # update the model
        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient
        # We use 100 as the maximum gradient norm because it is a common value used in the literature.
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)

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
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)


    def save_episode_results(self, max_score, total_score):
        '''
        This function saves the episode results.
        '''
        self.max_scores.append(max_score)
        self.total_scores.append(total_score)
    
    def save(self, filename):
        torch.save({
            'agent': self,
        }, filename)

    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename)
        agent = checkpoint['agent']
        return agent