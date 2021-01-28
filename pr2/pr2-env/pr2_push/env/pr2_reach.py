import os
import gym
import numpy as np 
import pr2_push
from gym import utils
from pr2_push.env import pr2_env

MODEL_XML_PATH = '/home/raghav/mujoco-pr2/pr2/pr2_reach.xml'

class PR2ReachEnv(pr2_env.PR2Env, utils.EzPickle):
    def __init__(self, reward_type = 'sparse'):
        initial_qpos = {
            "r_shoulder_pan_joint" : 0.15 ,
            "r_shoulder_lift_joint" : 0.65 , 
            "r_upper_arm_roll_joint": 0 , 
            "r_elbow_flex_joint" : -1 ,
            "r_forearm_roll_joint" : 0,
            "r_wrist_flex_joint" : -1 , 
            "r_wrist_roll_joint" : 0,
            # "puck_joint" : []
        }
        pr2_env.PR2Env.__init__(
            self,
            MODEL_XML_PATH,
            reward_type = reward_type,
            dist_threshold = 0.05,
            gripper_closed = True, 
            num_actions = 7, #Considering only the Joint position offsets
            initial_qpos = initial_qpos,
            target_range = 0.15, #default value for fetch_env = 0.15 m
            target_offset= 0.0, #default value in fetch_env = 0
            target_in_air = True, #Target will be in the air only for reach
            target_range_in_air = 0.5, #Only use when for Reach.py  
            object_present = False,
            object_range = 0.15, 
            n_substeps = 20)
        utils.EzPickle.__init__(self)

        """
        defining a class method for setting the properties of the 
        environment for dynamic randomization
        """

    def set_body_mass(self,dyn_prop_name,new_body_mass):
        bod_names = [n for n in self.sim.model.body_names if n.startswith('r_')]
        #number of bodies with name starting with "r_"
        num_bod_r = len(bod_names)
        # subtracting 5 gripper link indices which we are not randomizing
        num_bod = num_bod_r - 5 
        # print(num_bod)
        dyn_prop = getattr(self.sim.model, dyn_prop_name)
        body_id_arr = np.zeros(num_bod,dtype=np.int32)
        for i in range(num_bod):
            body_id_arr[i] = self.sim.model.body_name2id(bod_names[i])
            
        cnt = 1
        for j in range(len(dyn_prop)):
            # print("j:",j)
            if j==body_id_arr[cnt-1]:
            # print("cnt:",cnt)
                dyn_prop[j] = new_body_mass[cnt-1]
                cnt = cnt + 1
                if cnt > num_bod:
                    break
        self.sim.set_constants()
        # print(dyn_prop)


    def set_joint_friction(self,fri_attr_name,new_frictionloss):
        jnt_names = [n for n in self.sim.model.joint_names]
        prop_frictionloss = getattr(self.sim.model,fri_attr_name)
        length = 0
        for i in range(len(jnt_names)):
            jnt_qvel_arr = self.sim.data.get_joint_qvel(jnt_names[i])
            prop_frictionloss[i+length-1] = new_frictionloss[i+length-1]
    
    def set_joint_damping(self,damp_attr_name,new_damping):
        jnt_names = [n for n in self.sim.model.joint_names]
        prop_damping = getattr(self.sim.model,damp_attr_name)
        length = 0
        for i in range(len(jnt_names)):
            jnt_qvel_arr = self.sim.data.get_joint_qvel(jnt_names[i])
            prop_damping[i+length-1] = new_damping[i+length-1]

    # def set_control_gains(self,new_pid_gains):
        
    #     pass


    """
    Important Args Changes:
    before 27th January Initial Qpos: 
    np.array([0.15,0.65,0,-1.05,0,0,0])
    Changes on 27th January Initial QPos;
     np.array([0.15,-0.4,-3,-1.05,0,-1,0])
    """


    # np.array([0.15,0.65,0,-1.05,0,0,0])
    # 0.15,-0.4,-3,-1.05,0,-1,0
    # np.array([-0.6,0,-2,-1, 0,-1,0])
    # np.array([0.15,0.65,0,-1,0,-1,0])
    #Have to set the randomization for PID control coefficients

    ## getattr() is a built in python function/method that returns the 
    ## value of an attribute of an object.( object is an instance of a class)