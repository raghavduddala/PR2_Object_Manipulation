import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd 
from torch.autograd import Variable

"""
RDPG Implementation 
Previous action and the current state together should form the internal memory
which are updated as hidden state throughout an episode/trajectory

"""


class RDPGCritic(nn.Module):
    def __init__(self, dyn_parameters_dim, goal_dim, num_actions, num_states):
        super(RDPGCritic,self).__init__()
        self.dim_dyn_par = dyn_parameters_dim
        self.goal_dim = goal_dim
        self.action_dim = num_actions
        self.state_dim = num_states
        self.inp_dim1 = self.dim_dyn_par + self.goal_dim + self.action_dim + self.state_dim
        self.inp_dim2 = self.action_dim + self.state_dim
        self.fc1_1 = nn.Linear(self.inp_dim1, 400)
        self.fc1_2 = nn.Linear(self.inp_dim2,400)
        self.lstm = nn.LSTM( 400, 400, batch_first=True)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,1)

    def forward(self,dyn_parameters,goal,cur_action,state,prev_action, hidden_state = None):
        x_inp1 = torch.cat([dyn_parameters,goal,action,state],1)
        x_inp2 = torch.cat([state,prev_action],1)
        x_out1 = F.relu(self.fc1_1(x_inp1))
        x_out2 = F.relu(self.fc1_2(x_inp2))
        x_out_lstm, (hid_state,cell_state) = F.relu(self.lstm(x_out2, (hid_state,cell_state)))
        x_inp3 = torch.cat([x_out1,x_out_lstm],1)
        x_out = F.relu(self.fc2(x_inp3))
        x_out = F.relu(self.fc3(x_out))
        out = self.fc4(x_out)
        return out



class RDPGActor(nn.Module):
    def __init__(self,goal_dim, num_actions, num_states):
        super(RDPGActor,self).__init__()
        self.goal_dim = goal_dim
        self.state_dim = num_states
        self.action_dim = num_actions
        self.inp_dim1 = self.goal_dim + self.state_dim
        self.inp_dim2 = self.action_dim + self.state_dim
        self.fc1_1 = nn.Linear(self.inp_dim1,400)
        self.fc1_2 = nn.Linear(self.inp_dim2,400)
        self.lstm = nn.LSTM(400, 400, batch_first=True)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,7)

    def forward(self, goal, state, prev_action, hidden_state = None):
        x_inp1 = torch.cat([goal, state],1)
        x_inp2 = torch.cat([state,prev_action],1)
        x_out1 = F.relu(self.fc1_1(x_inp1))
        x_out2 = F.relu(self.fc1_2(x_inp2))
        x_out_lstm, (hid_state, cell_state)= F.relu(self.lstm(x_out2, (hid_state,cell_state)))
        x_inp3 = torch.cat([x_out1,x_out_lstm],1)
        x_out = F.relu(self.fc2(x_inp3))
        x_out = F.relu(self.fc3(x_out))
        out = torch.tanh(self.fc4(x_out))
        return out, hidden_state



#### Have to initialize and re-initialize the hidden state after each 
# trajectory 
#### Also , have to implement the weight initialization as mentioned 
# in the LSTM page