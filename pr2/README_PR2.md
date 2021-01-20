#Need to know how step works in sim.step and what are other essential methods in mujoco_py
#to control the robot
#Also, Simultaneosusly work on developing and playing around a basic pr2 environment

# ry to finish the communication work on the cartpole in Open AI gym 
# Sim2real extra papers that I found also bhairav mehta's paper with Dieter fox


#OPTION A: Open AI gym version using the code related to mujoco's equality weld constraint
# #OPtion B: Can try undestanding the code from ikpy and try to implement something like that 

# Even Joints are treated as constarints in the Mujoco for simulation 
# The inverse dynamics is calculated using the general equation of motions using the  Inertial
# mass matrix, jacobian, constaraint forces, applied joint torques, joint velocities and 
# coriolis compoenents of force 
# M*v_dot + c = tau + Jacobian_T*f_constraint - general equation of motion 

# v_dot = M_inv(tau + Jacobian_T*f_constraint) -- Forward dynamics
# tau = M*vdot + c - Jacobian_T*f_constraint -- Inverse Dynamics

# The M_inv in the Forward dynamics is calculated using the SPectral Decomposition(Diagonal
# Decomposition) Eigen Value Decomposition (Eigen value Decomposition can be applied because the Mass 
# Matrix is a square matrix with dimensions nv*nv (nv - degrees of freedom))

#### Option A basics from mujoco
# http://www.mujoco.org/forum/index.php?threads/controllig-force-for-proprioception-with-mocap.4185/
#Equality Constraint : Weld is used such that both the end-effector of the robot is joined with 
# a mocap body(which is not a physical entity and does not influence the physics of the model
# or the simulation). DIstance between the target and the mocap location is calculated. The 
# values of the mocap are changed in MJdata and the constraint solver in mujoco gives us the optmal
# acceleration required to position the end effector with the mocap body location
# we can control the amount if constraint forces in this regards with the solver paramters such as 
# "solimp" and "solref"
# It is solved using a  Constrained OPtimization Problem. It directly gives the joint accelerations/
# torques required to apply 
# It cannot be used without the simulation i.e we cannot get the required joint angles required
# as in a IK solver beacuse it directly solves the Inverse Dynamics Problem
# One has to implement a self-made IK Solver



#### 
# Try to implement by tomorrow to show something after completing the fetch_env and robot_env 

# Changes also have to be made to xml in regards to the (i)env set up and the correct name of 
# the bodies defined.(ii) and most importantly on the mocap weld joints!!! for solving the 
# equality constraint! for fetch I think I have to go through Robot Xml, Shared XML ... etc...
# 

# Function Implementation in the Gym Environment for Fetch:

# Action- the output from the Netwrok I believe for fetch is the X,Y, Z co-oridnates of the target location allong with the gripper positional difference
# Robot Env derives from the parent class Goal_env made for the purpose of HER 
# with sparse rewards just as the method mentioned in the dynamics randomization paper

# # Robot _env is the parent class where the main function/methods of 
    # Seed
    # Reset
    # Step
    # Viewer are implemented and some private methods that are used by the child class of any 
    # robot environment that we make (example - fetch_env) are also declared but not defined 


# Fetch_Env 
    # This has the definitions of the private methods that we were declared in the robot_env
    # and also uses the methods from utils and rotations classes defined in the 



# Important methods :


# _set_action: This contains the following methods from utils, the action input from the network is split first into the target position of the end-effector wrist and the gripper positional difference and is then augmented in the following order with the rotational quarternion (for a top-down grasp approach I guess) [pos_ctrl, quat_ctrl, grip_ctrl] THis is sent to the following methods explained below

    ## ctrl_set_action:  Splits the action into end-effector wrist position, quarternion and gripper_difference and uses only the gripper action which is a position difference that has to be attained by the position servo defined in the xml of the robot, this is nonetype for all the tasks except the "Pick and Place" where it is required to move the gripper and the position servos are defined in the pick_and_place robot xml

    ## mocap_set_action: splits the action into end-effector wrist position, quarternion and gripper_difference and uses the end-effector position and quaternion target difference (it is reshaped as (number_of_mocap_bodies,7*number_of_mocap_bodies)) 7 because the augmented action is an array of size 9 (splitting 7 exactly splits it to position, orientation target and the gripper position target). Uses the following function "reset_mocap2body_xpos"

    Also, uses mocap body to control the end effector. The Mocap is first placed with a weld constraint on the end effector and its position is shifted by  pos_delta, quat_delta....and then equality weld constraint optimization is solved as an optimization problem by mujoco to give the required accelerations/joint torques that can move the end-effector to the newly given target position and orientation

        ###reset_mocap2body_xpos: Before wanting to update the mocap body location at each time step with the pos_delta, quat_delta.. It is positioned to the current values of the bodies/objects it is welded to. in our case the end-effector



# _get_obs: Should see why site instead of object/link/body is considered, Site is considered 


# _viewer_setup: This method and the below method gets called in the render method of the parent class in robot_env, through the _get_viewer private method of the sam eparent class

# _render_callback: This is used to visualize whaterver object or site that we want to render in the simulation using the body/site_id by directly using the functions body_name2id /site_name2id  and the sim.model.site_pos[site_id]. I guess sim.forward is responsible for the rendering which carries the information about the above mentioned sites/bodies to render.


# _reset_sim: 


# _sample_goal: has two conditions, whether we have an object in the environment or not; if we have a object:



# _is_success: goal_distance method is used to find the distance between the last state of the episode with the desired goal configuration and checked against a specific tolerance.

# _env_setup: 



# _render: Calls the main render method from the parent class of robot_env


# What do the motion encoders from fetch robot gives us :
# THe code uses joint positions and velocities 
# for PR2 , we get joint acclerations too I guess

should check how differnt values in observation differ from each other....
like the grip_pos(from the site assigned) , gripper_state*from the robot's data) and 
grip_vel, gripper_vel





--> Implementing both RDPG and DDPG for verifying the results from Sim2real Transfer Paper.

--> 