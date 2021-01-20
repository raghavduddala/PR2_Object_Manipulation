import os
import gym
import pr2_push
from gym import utils
from pr2_push.env.pr2_env import PR2Env

"""
n-substeps: have to checked and implemented afterwards
target_offset : 
dist_threshold : 0.05 #defualt value should change this after looking into it
"""

####### Initial Joint configuration must be changed such that the robot
####### end-effector/grip is not in the air and actually near the table
# -0.655,-0.123,-0.8500,-0.567,0.25,0.5,0

MODEL_XML_PATH = '/home/raghav/mujoco-pr2/pr2/pr2.xml'

class PR2PushEnv(PR2Env, utils.EzPickle):
    def __init__(self,reward_type = "sparse"):
        initial_qpos = {
            "r_shoulder_pan_joint" : -0.655 ,
            "r_shoulder_lift_joint" : -0.123 , 
            "r_upper_arm_roll_joint": -0.8500 , 
            "r_elbow_flex_joint" : -0.567 ,
            "r_forearm_roll_joint" : 0.25 ,
            "r_wrist_flex_joint" : 0.5 , 
            "r_wrist_roll_joint" : 0.0 ,
            "puck_joint" : [1.25,0.53,0.75,1,0,0,0],
        }
        PR2Env.__init__(
            self,
            MODEL_XML_PATH,
            reward_type = reward_type,
            dist_threshold = 0.05,
            gripper_closed = True, 
            num_actions = 7, #Considering only the Joint position offsets
            initial_qpos = initial_qpos,
            target_range = 0.20, #default value for fetch_env = 0.15 m
            target_offset= 0.0, #default value in fetch_env = 0
            target_in_air = False, #Target will be in the air only for reach
            target_range_in_air = 0.5, #Only use when for Reach.py  
            object_present = True,
            object_range = 0.15, 
            n_substeps = 20)
        utils.EzPickle.__init__(self)
        #EzPickle is used when we use wrappers for C++/C MUjoco Files 
        # to create a serialization  of how the classes and their 
        # constructors are implemented or called.


        #Object range and Target range are used to initiate object position
        #  and goal position around a radius w.r.t the gripper position
