import torch 
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
from rdpg_model import RDPGActor, RDPGCritic
from memory_buffer import EpisodicBuffer, ReplayBuffer

class RDPGAgent:
    def __init__(self, base_env, dim_dyn_par, gamma, tau, actor_lr, critic_lr):
        self.dim_states = base_env.observation_space['observation'].shape[0] #  25D
        self.dim_actions = base_env.action_space.shape[0]  #7D
        self.dim_goal = base_env.observation_space['desired_goal'].shape[0] #3D
        self.dim_dyn_par = dim_dyn_par
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.mu = 0
        self.sigma = 0.1

        self.actor = RDPGActor(self.dim_goal, self.dim_actions, self.dim_states)
        self.target_actor = RDPGActor(self.dim_goal, self.dim_actions, self.dim_states)
        self.critic = RDPGCritic(self.dim_dyn_par, self.dim_goal, self.dim_actions, self.dim_states)
        self.target_critic = RDPGCritic(self.dim_dyn_par, self.dim_goal, self.dim_actions, self.dim_states)


    def sample_random_action(self):
        action = env.action_space.sample()
        return action

    def action_input_frm_network(self, state, prev_action, goal):
        state = Variable(torch.from_numpy(state).float())
        goal = Variable(torch.from_numpy(goal).float())
        prev_action = Variable(torch.from_numpy(prev_action).float())
        state = state.reshape(1,state.shape[0])
        goal = goal.reshape(1,goal.shape[0])
        prev_action = prev_action.reshape(1,prev_action.shape[0])

        action = self.actor.forward(goal,state, prev_action)
        action = action.detach().numpy()
        return action

    """
    Should see how to implement reinitializing the LSTM Hidden States
    """
    # def reset_lstm_states(self):
    #     pass


    def policy_update(self, replay_buffer, batch_size, episode_length):
        pass



    def save_model(self, output):
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(output))
    
    def load_model(self, output):
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def eval_model(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()