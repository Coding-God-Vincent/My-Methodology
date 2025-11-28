import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb  # 因為 cwd 為 "~/NCKU/Paper/My Methodology"，故使用相對匯入，匯入同資料夾的檔案。否則 python 會到 "~/NCKU/Paper/My Methodology/helpers" 中去找 helpers.py

'''
This file contains 2 models:
1. GDM (Construct by MLP) used to generate actions
2. Double Critic used to give GDM scores
'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''GDM (Actor)
* 3 inputs :
    1. state (s_t) : shape (batch_size, state_dim)
    2. time (t) : shape (batch_size)
    3. noise (x_t) : shape (batch_size, action_dim)
* 1 outputs :
    1. action (x_0) : shape (batch_size, action_dim)
'''

class GDM(nn.Module):
    def __init__(
        self, 
        state_dim,
        action_dim,
        hidden_dim= 256,
        t_dim= 16,  # use a 16-dimension tensor to represent a time by Sinusoidal Position embedding
        activation= 'mish'
    ):
        super().__init__()

        # activation function
        act = nn.Mish if activation == 'mish' else nn.ReLU

        # State Embedding Layer (output_shape (batch_size, hidden_dim))
        self.state_mlp = nn.Sequential(
            nn.Linear(in_features= state_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim)
        )

        # Time Embedding Layer (output_shape (batch_size, t_dim))
        self.time_mlp = nn.Sequential(
            # construct position_embedding tensor to represent t
            # shape (batch_size, t_dim)
            SinusoidalPosEmb(dim= t_dim),  
            nn.Linear(in_features= t_dim, out_features= t_dim*2),
            act(),
            nn.Linear(in_features= t_dim*2, out_features= t_dim)
        )
        
        # Decision-Making
        # input_shape (batch_size, hidden_dim + action_dim + t_dim)
        # output_shape (batch_size, action_dim)
        self.mid_mlp = nn.Sequential(
            nn.Linear(in_features= hidden_dim + action_dim + t_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= action_dim)
        )

    def forward(self, state, x_t, time):
        embedded_state = self.state_mlp(state)
        embedded_time = self.time_mlp(time)
        total_input = torch.cat([embedded_state, x_t, embedded_time], dim= 1)  # shape (batch_size, hidden_dim + action_dim + t_dim)
        return self.mid_mlp(total_input)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''Double Critic (Critic)
* 2 inputs:
    1. state (s_t) : shape (batch_size, state_dim)
    2. action (a_t) : shape (batch_size, action_dim)
* 2 kinds of outputs:
    1. outputs of 2 Q_networks : shape ((batch_size, 1), (batch_size, 1))
    2. min of 2 Q_networks : shape (batch_size, 1)
'''

class DoubleCritic(nn.Module):
    def __init__(
        self, 
        state_dim,
        action_dim,
        hidden_dim= 256,
        activation= 'mish'
    ):
        super().__init__()
        
        # activation function
        act = nn.Mish if activation == 'mish' else nn.ReLU

        # State Embedding Layer
        self.state_mlp = nn.Sequential(
            nn.Linear(in_features= state_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim)
        )

        # Q1
        self.Q_network1 = nn.Sequential(
            nn.Linear(in_features= hidden_dim + action_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= 1)
        )

        # Q2
        self.Q_network2 = nn.Sequential(
            nn.Linear(in_features= hidden_dim + action_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= 1)
        )

    def forward(self, state, action):
        embedding_state = self.state_mlp(state)
        total_input = torch.cat([embedding_state, action], dim= 1)  # shape (batch_size, hidden_dim + action_dim)
        return self.Q_network1(total_input), self.Q_network2(total_input)
    
    def q_min(self, state, action):
        return torch.min(*self.forward(state, action))

        