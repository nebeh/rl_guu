BATCH_SIZE = 64  
LEARNING_RATE = 0.001

import torch
import torch.optim as optim
import random
from model import QNetwork

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

device = torch.device("cuda" if use_cuda else "cpu")

from replay_buffer import ReplayMemory, Transition


class Agent(object):

    def __init__(self, n_states, n_actions, hidden_dim):
  
        
        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        
        self.mse_loss = torch.nn.MSELoss()
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)
        
        self.n_states = n_states
        self.n_actions = n_actions

        #  ReplayMemory: trajectory is saved here
        self.replay_memory = ReplayMemory(8224)
        

    def get_action(self, state, eps, check_eps=True):
   
        global steps_done
        sample = random.random()

        if check_eps==False or sample > eps:
           with torch.no_grad():
              
               return self.q_local(state.type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
           ## return LongTensor([[random.randrange(2)]])
           return torch.tensor([[random.randrange(self.n_actions)]], device=device) 


    def learn(self, experiences, gamma):
  
        
        if len(self.replay_memory.memory) < BATCH_SIZE:
            return;
            
        transitions = self.replay_memory.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
                        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        

        Q_expected = self.q_local(states).gather(1, actions)     

        Q_targets_next = self.q_target(next_states).detach().max(1)[0] 

        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        self.q_local.train(mode=True)        
        self.optim.zero_grad()

        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        loss.backward()
        self.optim.step()
        

