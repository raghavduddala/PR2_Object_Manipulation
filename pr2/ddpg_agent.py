import torch 
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
from ddpg_model import DDPGActor, DDPGCritic
from memory_buffer import EpisodicBuffer, ReplayBuffer
"""
Most of the code is similar to my DDPG implementation of the cartpole 
System
"""

class DDPGAgent:
    def __init__(self, base_env, dim_dyn_par, gamma, tau, actor_lr, critic_lr):
        self.dim_states = base_env.observation_space['observation'].shape[0] #  25D
        self.dim_actions = base_env.action_space.shape[0]  #7D
        print(base_env.action_space.high)
        print(base_env.action_space.low)
        self.dim_goal = base_env.observation_space['desired_goal'].shape[0] #3D
        self.dim_dyn_par = dim_dyn_par
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.mu = 0
        self.sigma = 0.1

        self.actor = DDPGActor(self.dim_goal, self.dim_states)
        self.target_actor = DDPGActor(self.dim_goal, self.dim_states)
        self.critic = DDPGCritic(self.dim_dyn_par, self.dim_goal, self.dim_actions, self.dim_states)
        self.target_critic = DDPGCritic(self.dim_dyn_par, self.dim_goal, self.dim_actions, self.dim_states)

        for target_parameters, main_parameters in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_parameters.data.copy_(main_parameters.data)

        for target_parameters, main_parameters in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_parameters.data.copy_(main_parameters.data)


    def action_input_frm_network(self, state, goal):
        state = Variable(torch.from_numpy(state).float())
        goal = Variable(torch.from_numpy(goal).float())
        state = state.reshape(1,state.shape[0])
        goal = goal.reshape(1,goal.shape[0])
        action = self.actor.forward(goal,state)
        # I guess we get action as a "Tensor Variable" which should 
        # be changed to numpy array 
        action = action.detach().numpy()
        return action


    def get_action_noise(self):
        noise = []
        for i in range(self.dim_actions):
            n = np.random.normal(self.mu, self.sigma)
            noise.append(n)
        noise = np.array(noise).reshape(self.dim_actions,)
        # print("noise action shape", noise.shape)
        return noise
 

    def eval_model(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def sample_random_action(self,env):
        # action = []
        # # random.seed(self.random_seed)
        # for i in range(self.dim_actions):
        #     action.append(random.uniform())
        # action = np.array(action)
        action = env.action_space.sample()
        return action


    def policy_update(self,replay_buffer, batch_size, episode_length):
        # #########First Implementation which is wrong########
        # sample_eps = replay_buffer.sample_episode_batch(batch_size)
        # # print(" sampled episodes:",sample_eps)
        # dim_row = batch_size*episode_length
        # state_batch = np.zeros((dim_row, self.dim_states))
        # action_batch = np.zeros((dim_row, self.dim_actions))
        # #Have to see what I put in the dimensions of the reward later
        # reward_batch = np.zeros(dim_row)
        # obs_batch = np.zeros((dim_row, self.dim_states))
        # goal_batch = np.zeros((dim_row, self.dim_goal))
        # done_batch = np.zeros(dim_row)
        # env_par_batch = np.zeros((dim_row,self.dim_dyn_par))
        
        # for i in range(batch_size):
        #     # print("i in agent:",i)
        #     state_arr,action_arr,reward_arr,obs_arr, _,done_arr = sample_eps[i].transitions_in_episodes()
        #     state_batch[i*episode_length:(i+1)*episode_length] = state_arr
        #     action_batch[i*episode_length:(i+1)*episode_length] = action_arr
        #     reward_batch[i*episode_length:(i+1)*episode_length] = reward_arr
        #     obs_batch[i*episode_length:(i+1)*episode_length] = obs_arr
        #     # print("Goal from sampled episodes",sample_eps[i].desired_goal())
        #     goal_batch[i*episode_length:(i+1)*episode_length] = sample_eps[i].desired_goal()
        #     done_batch[i*episode_length:(i+1)*episode_length] = done_arr
        #     env_par_batch[i*episode_length:(i+1)*episode_length] = sample_eps[i].environment_parameters()
            # print("environment from sampled episodes",sample_eps[i].environment_parameters())


            # Args for critic forward : env_parameters,goal,action,state
        ####################Second IMplementation
        sampled_eps_ids = replay_buffer.sample_episode_batch(batch_size)
        episodes_sampled = replay_buffer.get_sampled_episodes(sampled_eps_ids)
        state_batch = []
        action_batch = []
        reward_batch = []
        obs_batch = []
        goal_batch = []
        done_batch = []
        env_par_batch = []
        assert(len(episodes_sampled)) == batch_size
        for i in range(batch_size):
            experience = episodes_sampled[i].sample_random_experience()
            state_batch.append(experience[0][0])
            action_batch.append(experience[0][1])
            reward_batch.append(experience[0][2])
            obs_batch.append(experience[0][3])
            done_batch.append(experience[0][5])
            goal_sampled = episodes_sampled[i].desired_goal()
            goal_batch.append(goal_sampled)
            env_par_sampled = episodes_sampled[i].environment_parameters()
            env_par_batch.append(env_par_sampled)
        ##########################################
        state_batch = torch.Tensor(state_batch)
        action_batch = torch.Tensor(action_batch)
        reward_batch = torch.Tensor(reward_batch)
        obs_batch = torch.Tensor(obs_batch)
        goal_batch = torch.Tensor(goal_batch)
        done_batch = torch.Tensor(done_batch)
        env_par_batch = torch.Tensor(env_par_batch)

        critic_value = self.critic.forward(env_par_batch,goal_batch,action_batch,state_batch)
        # print("Critic value shape", critic_value.shape)
        action_plus_batch = self.target_actor.forward(goal_batch,state_batch)
        critic_target_value = self.target_critic.forward(env_par_batch, goal_batch,action_plus_batch.detach(),state_batch)
        # print("crtitic target value shape", critic_target_value.shape)
        
        ###Original Implementation of y_value 
        # y_value = torch.reshape(reward_batch,(dim_row,1)) + (1-torch.reshape(done_batch,(dim_row,1)))*self.gamma*critic_target_value
        ### Changed IMplementation of y_value
        y_valuepart1 = torch.reshape(reward_batch,(batch_size,1))
        # print("y_valuepart1 shape", y_valuepart1.shape)
        y_valuepart2 = self.gamma*critic_target_value
        # print("y_valuepart2 shape",y_valuepart2.shape) (1-torch.reshape(done_batch,(batch_size,1)))
        y_value = y_valuepart1 + y_valuepart2
        # print("y_value shape", y_value.shape)
        # y_value = []
        # for j in range(dim_row):
        #     if done_batch[j]:
        #         y_value.append(reward_batch[j])
        #     else:
        #         y_value.append(reward_batch[j]+ self.gamma*critic_target_value[j])

        # y_value = torch.Tensor(y_value)
        # y_value = torch.reshape(y_value,(dim_row,1))

        loss_criterion = nn.L1Loss()
        critic_loss = loss_criterion(critic_value,y_value).mean()
        # print("Critic loss value", critic_loss)

        #Initialising the Optimizer for Critic
        optim.Adam(self.critic.parameters(),lr = self.critic_lr).zero_grad()
        #Calculating Backward Loss and then performing one step of optimization
        critic_loss.backward()
        optim.Adam(self.critic.parameters(),lr = self.critic_lr).step

        action_policy_batch = self.actor.forward(goal_batch,state_batch)
        policy_loss = -self.critic.forward(env_par_batch,goal_batch,action_policy_batch,state_batch).mean()
        # print("Policy LOss Value", policy_loss)
        #Initialising the Optimizer for Actor
        optim.Adam(self.actor.parameters(),lr = self.actor_lr).zero_grad()
        #Calculating Backward Loss and then performing one step of optimization
        policy_loss.backward()
        optim.Adam(self.actor.parameters(),lr = self.actor_lr).step



        #Weight Updates of Target Networks
        #Using Polyak Averaging from the paper 
        #for this we will be suing .copy_() method from pytorch 

        for target_parameters, main_parameters in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_parameters.data.copy_(main_parameters.data*self.tau + target_parameters.data*(1.0-self.tau))

        for target_parameters, main_parameters in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_parameters.data.copy_(main_parameters.data*self.tau + target_parameters.data*(1.0-self.tau))

        return critic_loss, policy_loss
#  State-dict is a python dictionary object that maps each layer with its
#  parameter tensor
# load_state_dict/ save_state_dict are methods used for saving and loading 
# the state_dict

       
    def save_model(self, output):
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(output))
    
    def load_model(self, output):
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    
