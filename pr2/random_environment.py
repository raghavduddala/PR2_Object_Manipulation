import numpy as np 
import gym
import random

import pr2_push


class RandomizeEnvironment:
    def __init__(self,env_to_randomize,dyn_par_ranges):
        self.env_to_randomize = env_to_randomize
        self.dyn_par_ranges = dyn_par_ranges
        self.len_dyn_par = 0
        self.env_param = []

    def sample(self):
        low_mass = self.dyn_par_ranges[0][0]
        high_mass = self.dyn_par_ranges[0][1]
        low_friction = self.dyn_par_ranges[1][0]
        high_friction = self.dyn_par_ranges[1][1]
        low_damping = self.dyn_par_ranges[2][0]
        high_damping = self.dyn_par_ranges[2][1]
        random_mass = []
        random_friction = []
        random_damping = []
        env_param = []
        # random.seed(self._random_seed)
        # for i in range(9):
        #     random_mass.append(random.uniform(low_mass, high_mass))
        # for j in range(11):
        #     random_friction.append(random.uniform(low_friction,high_friction))
        #     random_damping.append(random.uniform(low_damping,high_damping))
        # random_mass = [round(k,8) for k in random_mass]
        # self.env_param.append(random_mass)
        # random_friction = [round(i, 8) for i in random_friction]
        # self.env_param.append(random_friction)
        # random_damping = [round(j, 8) for j in random_damping]
        # self.env_param.append(random_damping)
        # self.env_param = np.array(self.env_param)

        for i in range(9):
            value_mass = round(random.uniform(low_mass, high_mass),8)
            env_param.append(value_mass)
            random_mass.append(value_mass)
        for j in range(11):
            value_fricloss = round(random.uniform(low_friction, high_friction),8)
            env_param.append(value_fricloss)
            random_friction.append(value_fricloss)
        for k in range(11):
            value_damping = round(random.uniform(low_damping,high_damping),8)
            env_param.append(value_damping)
            random_damping.append(value_damping)
    
        # self.env_param = np.array(self.env_param)
        # print("frictionloss from rand_env.py:",random_friction) 
        # print("damping from rand_env.py:", random_damping) 
        # print("Mass from rand_env.py:", random_mass) 
        self.len_dyn_par= len(random_mass) + len(random_friction) + len(random_damping)
        self.env = gym.make(self.env_to_randomize)
        self.env.set_body_mass("body_mass",random_mass)
        self.env.set_joint_friction("dof_frictionloss",random_friction)
        self.env.set_joint_damping("dof_damping",random_damping)
        self.env_param = env_param

    def env_n_parameters(self):
        return self.env, self.len_dyn_par, self.env_param

    def close_env(self):
        self.env.close()

"""
# TODO:
1. Have to decide if the new random mass array must be given to the 
above method as an input or generated with in the above method 

"""