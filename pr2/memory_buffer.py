from collections import deque
import numpy as np 
import random

"""
Implementing a Double ended Queue of class objects 
The class objects are episodes containing lists of States, Actions, Reward
and information about the terminal state of an episode.
"""


class EpisodicBuffer:
    def __init__(self, goal, env_params, max_length):
        self.max_length = max_length
        self.episode = []
        self.des_goal = goal
        self.env_params = env_params

    def add_episode_step(self,state,action,reward,obs,ach_goal,done):
        experience_transition = (state,action,reward,obs,ach_goal,done)
        self.episode.append(experience_transition)
        # print(self.episode)

    def transitions_in_episodes(self):
        state_exp = []
        action_exp = []
        reward_exp = []
        obs_exp = [] 
        ach_goal_exp = []
        eps_end_exp  = []
        goal_exp = []
        env_params_exp = []


        for trans_step in self.episode:
            state,action,reward,obs,ach_goal,done = trans_step
            state_exp.append(state)
            action_exp.append(action)
            reward_exp.append(reward)
            obs_exp.append(obs)
            ach_goal_exp.append(ach_goal)
            eps_end_exp.append(done)

        
        state_exp = np.array(state_exp)
        action_exp = np.array(action_exp)
        reward_exp = np.array(reward_exp)
        obs_exp = np.array(obs_exp)
        ach_goal_exp = np.array(ach_goal_exp)
        eps_end_exp = np.array(eps_end_exp)

        return state_exp,action_exp,reward_exp,obs_exp,ach_goal_exp,eps_end_exp

    # def sample_random_experience(self):

    #     return experience
    
    
    def desired_goal(self):
        return self.des_goal

    def environment_parameters(self):
        return self.env_params


       



class ReplayBuffer:
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = self.buffer_size)
        self.cur_buffer_length = 0

    def add_episode(self,episode):
        if self.cur_buffer_length>=self.buffer_size:
            self.buffer.popleft()

        self.buffer.append(episode)
        # print(self.buffer)

    def sample_episode_batch(self,batch_size):
        if batch_size > self.cur_buffer_length:
            print("Sampling is Invalid")

        range_buffer = range(0, self.cur_buffer_length)
        sampled_ids = random.sample(range_buffer, batch_size)
        # print("Sampled_Episodes for Optimization",sampledss)
        #### Original IMplementation
        # sampleds = random.sample(self.buffer, batch_size)
        return sampled_ids


    def len_buffer(self):
        self.cur_buffer_length = len(self.buffer)
        return self.cur_buffer_length



