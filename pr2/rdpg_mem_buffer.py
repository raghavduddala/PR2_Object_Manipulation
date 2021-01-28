from collections import deque
import numpy as np 
import random

"""
Implementing a Double ended Queue of class objects 
The class objects are episodes containing lists of States, Actions, Reward
, Previous actions and information about the terminal state of an episode.
"""

class RDPG_EpisodicBuffer:
    def __init__(self, goal, env_params, max_length):
        self.max_length = max_length
        self.episode = []
        self.des_goal = goal
        self.env_params = env_params

    def get_trajectory_episode(self):
        pass

# Changes have to be made accordingly for the add episode step to 
#   be careful about the previosu action as input to the network
    def add_episode_step(self,state,action,reward,obs,ach_goal,done):
        experience_transition = (state,action,reward,obs,ach_goal,done)
        self.episode.append(experience_transition)


    def desired_goal(self):
        return self.des_goal

    def environment_parameters(self):
        return self.env_params



class RDPG_ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = self.buffer_size)
        self.cur_buffer_length = 0

    def sample_episode_batch(self, batch_size):
        if batch_size > self.cur_buffer_length:
            print("Sampling is Invalid")

        range_buffer = range(0, self.cur_buffer_length)
        sampled_ids = random.sample(range_buffer, batch_size)
        return sampled_ids
    

    def get_sampled_episodes(self,list_indx):
        sampled_episodes = []
        for i in range(len(list_indx)):
            idx = list_indx[i]
            sampled_episodes.append(self.buffer[idx])
        
        return sampled_episodes

    def len_buffer(self):
        self.cur_buffer_length = len(self.buffer)
        return self.cur_buffer_length