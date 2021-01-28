import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd 
from torch.autograd import Variable

"""
According to the Sim2real paper, for DDPG only the state and goal are 
passed into the POLICY Network and 
only the combined vector(randomized parameters, goal, cur_action) and state
are passed into the CRITIC Network
No additional state and action concatenation is done in both 
Critic and Actor NN
"""

class DDPGCritic(nn.Module):
    def __init__(self,dyn_parameters_dim,goal_dim,num_actions,num_states):
        super(DDPGCritic, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_dyn_par = dyn_parameters_dim
        self.goal_dim = goal_dim
        self.fc1_inp_dim = self.num_dyn_par + self.goal_dim + self.num_actions
        self.fc1 = nn.Linear(self.num_states + self.fc1_inp_dim,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,1)


    def forward(self,dyn_parameters,goal,cur_action,state):
        """
        Args: N - Batch Size
        Dynamic paramters:
        goal: 
        action: Nx7 
        state: Nx25
        """

        x1 = dyn_parameters
        x2 = goal
        x3 = cur_action
        x4 = state

        x1_inp = torch.cat([x1,x2,x3,x4],1) 
        x1_out = F.relu(self.fc1(x1_inp))
        x2_out = F.relu(self.fc2(x1_out))
        x3_out = F.relu(self.fc3(x2_out))
        out = self.fc4(x3_out)
        return out


class DDPGActor(nn.Module):
    def __init__(self,goal_dim,num_states):
        super(DDPGActor,self).__init__()
        self.num_states = num_states
        self.goal_dim = goal_dim
        self.fc1 = nn.Linear(self.goal_dim + self.num_states,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,7)

    def forward(self,goal,state):
        """
        Args: N - Batch Size
        State: Nx25
        Goal: Nx3
        """
        # print(goal.shape)
        # print(state.shape)
        x1_inp = torch.cat([goal,state],1)
        x1_out = F.relu(self.fc1(x1_inp))
        x2_out = F.relu(self.fc2(x1_out))
        x3_out = F.relu(self.fc3(x2_out))
        out = torch.tanh(self.fc4(x3_out))
        return out






# to concatenate two tensors, generally we have the first dimension as
# the batch_size, that is why we concatenate t3 = torch.cat((t1,t2),1)

#Version on January 11, deleted the num_actions(action dimensions in actor 
# network as no longer required)

#Version 18th January - Changing network dimensions from 128 each to 400,
# 300 and then so on.