import os
import numpy as np 
import gym
import pr2_push

from collections import deque


# # exp = "PR2Push-v0"
# exp = "PR2Reach-v0"


# env = gym.make(exp)


# init_obs_dict = env.reset()
# des_goal = init_obs_dict['desired_goal']
# print("Desired goal from test.py", des_goal)
# ach_goal = init_obs_dict['achieved_goal']
# print("Achieved goal from test.py", ach_goal)
# obs = init_obs_dict['observation']
# print("Obs from test.py", obs)
# obs_list = []
# for _ in range(1000):
#     env.render()
#     obs, reward, done, info = env.step(env.action_space.sample())
#     obs_list.append(obs)
# env.close()