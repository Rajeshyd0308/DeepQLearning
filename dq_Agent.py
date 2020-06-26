import numpy as np
import random
from collections import namedtuple, deque

from model import network

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent():
    
    def __init__(self, state_size, action_size):
        """Initializes some key parameters"""
        """

        Params
        ======
            state_size (int): State size
            action_size (int): Action vector size
        """
        self.state_size = state_size
        self.action_size = action_size
        
        hidden_layers = [64,64,64] # Hidden layers list
        self.qlocal = network(self.state_size, self.action_size, hidden_layers).to(device) # Local NN, updated for every 4 iterations
#         print("here")
        self.qtarget = network(self.state_size, self.action_size, hidden_layers).to(device) # Target NN, softupdated
        
        self.optimizer = optim.Adam(self.qlocal.parameters(), lr=LR)
        
        self.buffer = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE) # Prioritized memory buffer
        self.t_step = 0 # Time step counter
        
    def act(self, state, eps=0.0):
        """Given the state, returns an action, based on our current parameters in qlocal"""
        """

        Params
        ======
            state (numpy array): state vector
            eps (float): Epislon
        """
        # Episilon greedy policy
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # Convert numpy state vector to torch float tensor.
        self.qlocal.eval()      # We do not need to have drop out when we are not training
        with torch.no_grad():        # We are not traning so we turn off gradients
            action_values = (self.qlocal.forward(state)).cpu().numpy()
        self.qlocal.train()     # Turning dropout on afterwards so we can train
        if random.random()>eps:  # Follow eps greedy policy,
            return np.argmax(action_values)
        else:  # select an action randomly
            return random.choice(np.arange(self.action_size))
        
    
    def step(self, state, action, reward, next_state, done):
#         e = ()        
        """Adds state, action, reward, nextstate, done to buffer, and sets error to 1, which is maximum. So that this SARSA pair can be 
        explored"""
        """

        Params
        ======
            state (numpy array): state vector
            action (int): action value (0-action_size)
            reward (int): reward
            next_state (numpy array): next state vector
            done (int): 1 if we reach the terminal state
        """
        self.buffer.add(state, action, reward, next_state, done) # Adding experiences
        self.t_step +=1
        if self.t_step%UPDATE_EVERY==0 and len(self.buffer)>BATCH_SIZE: # Training of NN is run for UPDATE_EVERY (4) timesteps
            self.t_step=0 # Reset time step
            sample_experiences = self.buffer.sample() # Priority sample of experiences
            self.learn(sample_experiences) # Forward prop and SGD
    
    def learn(self, experiences):
        """Training of our NN is done hear training includes- forward prop, backprop and update of gradients."""
        """

        Params
        ======
            experiences (torch tensor): tuple of states,...
        """
        # Everything in this function is a torch tensor
        states, actions, rewards, next_states, dones = experiences 
        Qvals = self.qlocal.forward(states).gather(1,actions) # forward prop of states torch tensor, size is self.batch_size, 37
#         Qtargets = self.qtarget.forward(next_states).detach().max(1)[0].unsqueeze(1) # Frwd prop inorder to select best action available from next state
        # We select action using the target network, and take the q value from Local network of the corresponding action.
        # This is Double DQN
        action_max = torch.argmax(self.qtarget.forward(next_states).detach(), dim=1).unsqueeze(1)  
        Qtargets = self.qlocal.forward(next_states).detach().gather(1,action_max)
        Gt = rewards+(GAMMA*Qtargets*(1-dones)) # Temporal difference target
        loss = F.mse_loss(Gt, Qvals) # This loss is backpropageted

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qlocal, self.qtarget, TAU) # Soft update of target parameters        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
class ReplayBuffer:
    """Replay buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # Buffer
        
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
#         self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
#     def update(self, sample_index, error):
#         """Updates TD Error for each experience which was previously sampled"""
#         delta = np.abs(error.data.cpu().numpy())
#         j= 0
#         for i in sample_index:
#             self.priorities[i] = float(delta[j]+self.e)
#             j+=1
#         self.priorities[sample_index]
    
    def sample(self):
        """ Sample a batch of experiences from memory"""

        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)