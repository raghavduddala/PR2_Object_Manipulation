import gym
import numpy as np

from gym.envs.robotics import rotations, robot_env
from pr2_push.env import robot_utils

"""
Using the Robot_env since it can be easily used with HER and the RDPG
since the observation used in robot_env module from goal_env has both achieved goal 
and desired goal

Object Distance threshold in _reset_sim for object position initiation 
is manually set to 10 cm or 0.1 m 

The goal is generated twice. Once during the intilisation of the 
environment and the other when the environment is reset( the_reset_sim 
function gets called which calls sim.forward which performs some action

"""


class PR2Env(robot_env.RobotEnv):
    """
    Super class for all the other pr2 env setups
    Using robot_env class as parent class since we need the goal_env
    for implementing HER with different goal position randomizations
    """

    def __init__(
        self, model_path, reward_type, dist_threshold, 
        gripper_closed,num_actions,initial_qpos,target_range,target_offset,
        target_in_air, target_range_in_air, object_present, object_range,n_substeps):

        """
        Args 

        model_path  : The file path for the XML Model 
        reward_type : Sparse or Dense Rewards
        init_qpos   : Initial Joint positions of robot joints and/or object position
        num_actions : Dimension of the action space --7 (7 DOF Joint target offsets)
        self.sim    : Variable defined and declared in the parent "robot_env" class
        """
        self.gripper_closed = gripper_closed
        self.dist_threshold = dist_threshold
        self.reward_type = reward_type
        self.object_present = object_present
        self.target_range = target_range
        self.target_offset = target_offset
        self.target_in_air = target_in_air
        self.target_range_in_air = target_range_in_air 
        self.object_range = object_range
        super(PR2Env, self).__init__(model_path = model_path,n_substeps = n_substeps,n_actions = num_actions,initial_qpos = initial_qpos)
        

    def compute_reward(self, achieved_goal, goal, info):
        """
        compute_reward method is declared in the GoalEnv Class
        which is parent class to RobotEnv and then PR2Env
        GoalEnv > RobotEnv > PR2Env
        """
        # reward is basically negative
        if achieved_goal.shape == goal.shape:
            dist = np.linalg.norm(achieved_goal-goal,axis=-1) # dimensions of the goal is 3 (Cartesian coordinates)
            if self.reward_type == 'sparse':
                return -(dist > self.dist_threshold).astype(np.float32)
            else:
                return -dist


    def _step_callback(self):
        """
        Mandatory Callback Function to constrain the Gripper to remain closed after 
        each step in simulation
        """
        #Should figureout if I can constraint them through control only (using _set_actions)
        #Setting all the finger related joints according to defined xml 
        if self.gripper_closed:
            self.sim.data.set_joint_qpos("r_gripper_l_finger_joint",0)
            self.sim.data.set_joint_qpos("r_gripper_r_finger_joint",0)
            self.sim.data.set_joint_qpos("r_gripper_l_finger_tip_joint",0)
            self.sim.data.set_joint_qpos("r_gripper_r_finger_tip_joint",0)
            self.sim.forward() 
            # sim.forward does not integrate in time should see if I can use this.
            # Does not advance the simulation state by one time-step 

    def _set_action(self,action):
        # Action Dimensions: (7)- (Output from NN in Push task) Joint Angles without gripper (using _step_callback)
        # Action Dimensions: (11)- (According to XML) Joint Angles with gripper (w/o _step_callback being called in robot_env class)
        
        # Gripper Joint is prismatic with Joint Limits : 86mm -- 0 mm according to PR2 Manual from Clearpath
        # https://www.clearpathrobotics.com/assets/downloads/pr2/pr2_manual_r321.pdf
        
        assert action.shape == (7,)
        action = action.copy()  # Unsure if action value changes // can use action.copy
        joint_pos_ctrl = action[:4]
        passive_pos_ctrl = [0., -1., 0.]
        #Rescaling joint angles by a facto of 1/20
        # joint_pos_ctrl *= 0.05 # Limiting the max joint angles as in fetch_env to see
        #any change in the applied torque
        # grip_ctrl = action_ctrl[7:] #Since no Gripper Control output is given by NN in push task
        action = np.concatenate([joint_pos_ctrl, passive_pos_ctrl])
        # robot_utils.set_joint_action(self.sim,joint_pos_ctrl) 
        robot_utils.set_joint_action(self.sim,action)


    def _is_success(self,achieved_goal,goal):
        # returns "1" for true "0" for false
        assert achieved_goal.shape == goal.shape
        dist = np.linalg.norm(achieved_goal-goal,axis=-1) #Have to implement with the correct shape 
        return (dist < self.dist_threshold).astype(np.float32)

    def _get_obs(self):
        # 1.Also Observation should follow the Sim2real paper observation
        # dimensions
        # 2.Have to return three things: observation, achieved goal, desired goal
        gripper_pos = self.sim.data.get_site_xpos('grip_site')
        #My understanding of mutliplying with dt is such that the 
        # average velocity over the n_substeps is considered 
        dt = self.sim.nsubsteps*self.sim.model.opt.timestep
        gripper_velp = self.sim.data.get_site_xvelp('grip_site')*dt
        robot_qpos, robot_qvel = robot_utils.robot_obs(self.sim)
        # print("robot qpos size", robot_qpos.shape)
        # print("robot qvel size", robot_qvel.shape)
        # print("gripper site pos", gripper_pos.shape)
        if self.object_present:
            #Object position and rotation
            object_pos = self.sim.data.get_site_xpos('object0')
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            #Object linear and angular velocities
            object_velp = self.sim.data.get_site_xvelp('object0')*dt
            object_velr = self.sim.data.get_site_xvelr('object0')*dt
        else:
            #np.zeros(0) -- Initializes an empty array 
            object_pos = object_rot = object_velp = object_velr = np.zeros(0)

        if not self.object_present:
            achieved_goal = gripper_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([robot_qpos, 
                            robot_qvel,
                            gripper_pos,
                            # gripper_velp,     
                            object_pos.ravel(),
                            object_rot.ravel(),
                            object_velp.ravel(),
                            object_velr.ravel()])
        return {
            'observation' : obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal' : self.goal.copy(),
        }

    def _viewer_setup(self):
        #WHat to mark and see in the simulation rendering
        # body_id = self.sim.model.body_name2id("")
        #defualt values from fetch_env are 2.5, -14, 132
        self.viewer.cam.distance = 3
        self.viewer.cam.elevation = -14
        self.viewer.cam.azimuth = 140

    def _render_callback(self):
        # visualize target.
        #Should Figure out how the sites_offset is used
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        #Visualize the robot gripper
        #  site_id_grip = self.sim.model.site_name2id('grip_site')
        # self.sim.model.site_pos[site_id_grip]
        self.sim.forward()


    def _reset_sim(self):
        # to reset the simulation used after initiating the environment
        # or after using gym.make
        # self.initial_state = copy.deepcopy(self.sim.get_state()) defined
        # in the main robot_env class that gets called first where we load 
        # the self.sim = MjSim(Model)
        self.sim.set_state(self.initial_state)

        if self.object_present:
            #First setting the (x,y) coordinates of the object as the 
            # gripper xpos so that the object position can be set relative 
            # or w.r.t to the gripper position
            object_xpos = self.initial_gripper_xpos[:2]
            #Then in a loop setting the object position by randomizing with a 
            #standard deviation of object_range such that the final position 
            # of the object is with in 10 cm from gripper
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.object_range, self.object_range)
            object_qpos = self.sim.data.get_joint_qpos('puck_joint')
            assert object_qpos.shape ==(7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('puck_joint',object_qpos)

        self.sim.forward()
        return True


    def _sample_goal(self):
        #Two conditions - whether the object is present or not
        if self.object_present:
            #Considering only the 3-D cartesian Co-ordinates
            #and (plus, minus) the standard deviation of taregt range 
            #w.r.t to the initial_gripper_position
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,self.target_range)
            #if there is a target_offset add even that i.e the mean is shifted 
            # or bias is added in otherwords
            goal = goal + self.target_offset
            #considering height offset such that the gripper position
            #when the goal is achieved is equal to object's height
            goal[2] = self.height_offset
            if self.target_in_air  and self.np.random.uniform()<self.target_range_in_air:
                goal[2] += self.np_random.uniform(0,(self.target_range_in_air-0.05))
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,self.target_range)
        #or should I be using goal.copy ()
        # print("Goal co-ordinates:",goal)
        return goal.copy()


    def _env_setup(self,initial_qpos):
        #Setting up the initial joint configurations to a known 
        # or given pose in PR2PushEnv/PR2ReachEnv Child class
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name,value)
        
        # self.sim.forward()

    
        ##After setting up the environment -- to move it for some timesteps
        for _ in range(5):
            self.sim.step()
        
        #getting the gripper position after moving the robot for 10-15 steps
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grip_site').copy()
        # print("Gripper Cartesian Co-ords",self.initial_gripper_xpos)
        # print("initial_gripper_xpos_initialized")
        if self.object_present:
            #Height offset so that the goal can be set accordingly
            #considering the object height such that the end gripper
            #height from table equals the object's height
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
    

    def render(self, mode='human', width=500, height=500):
        return super(PR2Env, self).render(mode, width, height)



## nicrusso7/rex-gym
## jr-robotics/robo-gym
## https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa

#Lets not consider gripper actuation at all right now, 
#will get the End-effector pose attempt learning problem first 