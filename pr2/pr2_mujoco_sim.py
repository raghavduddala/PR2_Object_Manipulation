#!/usr/bin/python

#have to see what modder does?
#Look into MjSim and MjViewer methods and also "model" methods

from mujoco_py import load_model_from_path, MjSim, MjViewer
# from mujoco_py.modder import TextureModder
import os
import numpy as np 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET



# model = load_model_from_path("pr2/pr2.xml")
# # print(model.nbody)

# sim = MjSim(model)
# #200,1000,1000,1000,500,1000,

# # Best values : 20,500,40,40,20,20,20

# # for i in range(sim.model.nbody):
# #     print("Body ID:",i,",","Body Name:",sim.model.body_id2name(i))

# # for j in range(sim.model.njnt):
# #     print("Joint ID:",j,",","Joint Name:",sim.model.joint_id2name(j),",","Joint Limits",sim.model.jnt_range[j])

# # for k in range(len(sim.data.ctrl)):
# #     print("Actuator ID:",k,",","Actuator Name:",sim.model.actuator_id2name(k),",","Range:",sim.model.actuator_ctrlrange[k])

# # # print(sim.data.ctrl)
# # print("Total Constraints:", sim.data.nefc)
# # print("Constraint Jacobian shape:", sim.data.efc_J.shape)
# # print("Total Degrees of Freedom:", sim.model.nv)

# # #It gives us the ID Of the joint being controlled by the actuator defined in a serial manner
# # print(sim.model.actuator_trnid)


# # Finger actuators have not been defined - which I should do
# # Each motor  should have different controller settings since the gains whould be different 
# # for achieving different types of tasks. Have to look into coding up a PID controller 
# # for each of the joint separately


# viewer = MjViewer(sim)
# # modder = TextureModder(sim)

# sim_horizon = 500

# sim_state = sim.get_state()


# # sim.data.ctrl[0] = -0.655
# # sim.data.ctrl[1] = -0.123
# # sim.data.ctrl[2] = -0.8500
# # sim.data.ctrl[3] = -0.567
# # sim.data.ctrl[3:] = 0

# sim.data.ctrl[:] = 0

# # print(len(sim.data.qpos))
# N = len(sim.data.ctrl)
# joint_pos = np.zeros((sim_horizon,N))
# joint_ref = np.zeros((sim_horizon,N))
# ext_jnt_force = np.zeros_like(joint_ref)
# #repeat indefinitely
# # while True:
# #     #set simulation to initial state
# #     sim.set_state(sim_state)
# #     # sim.data.qpos[20] = 1
# #     #for the entire simulation horizon
# #     for i in range(sim_horizon):
# #         sim.step()
# #         viewer.render()

# for i in range(sim_horizon):
#     sim.step()
#     joint_indx = []
#     for j in range(len(sim.data.ctrl)):
#         idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[j,0]]
#         joint_pos[i,j] = sim.data.qpos[idx]
#         joint_ref[i,j] = sim.data.ctrl[j]
#         ext_jnt_force[i,j] = sim.data.qfrc_applied[idx]
#     viewer.render()

# print("Joint forces:", ext_jnt_force[sim_horizon-1,:])
# std_err = abs(joint_pos[sim_horizon-1,:]- joint_ref[sim_horizon-1,:])
# print("Steady State Error:",std_err)


# num_steps = np.arange(sim_horizon)
# # plt.plot(num_steps,joint_pos[:,0],num_steps,joint_pos[:,1],num_steps,joint_pos[:,2],num_steps,joint_pos[:,3])
# fig,ax = plt.subplots(4,2)
# ax[0,0].plot(num_steps,joint_pos[:,0],num_steps,joint_ref[:,0])
# ax[0,0].set_title('Shoulder Pan')
# ax[0,1].plot(num_steps,joint_pos[:,1],num_steps,joint_ref[:,1])
# ax[0,1].set_title('Shoulder Lift')
# ax[1,0].plot(num_steps,joint_pos[:,2],num_steps,joint_ref[:,2])
# ax[1,0].set_title('Upper arm roll')
# ax[1,1].plot(num_steps,joint_pos[:,3],num_steps,joint_ref[:,3])
# ax[1,1].set_title('Elbow flex')
# ax[2,0].plot(num_steps,joint_pos[:,4],num_steps,joint_ref[:,4])
# ax[2,0].set_title('Lower arm roll')
# ax[2,1].plot(num_steps,joint_pos[:,5],num_steps,joint_ref[:,5])
# ax[2,1].set_title('Wrist flex')
# ax[3,0].plot(num_steps,joint_pos[:,6],num_steps,joint_ref[:,6])
# ax[3,0].set_title('Wrist roll')
# # plt.xlabel("time")
# # plt.ylabel("Joint Angle in Radians")

# for i in ax.flat:
#     i.set(xlabel='Time', ylabel='Radians')
# for j in ax.flat:
#     j.label_outer()
# plt.legend(["joint angle","Reference"])
# # plt.legend(["Shoulder Pan","Shoulder Lift","Upper arm roll", "Elbow FLex"])
# plt.show()

#pid position controller for 
######################### PD Controller ###################
MODEL_PATH = "/home/raghav/mujoco-pr2/pr2/pr2_exp.xml"
model1 = load_model_from_path(MODEL_PATH)

sim1 = MjSim(model1)
viewer1 = MjViewer(sim1)

sim_horizon = 5000

dt = sim1.model.opt.timestep

print(sim1.model.joint_names)
for j in range(sim1.model.njnt):
    print("Joint ID:",j,",","Joint Name:",sim1.model.joint_id2name(j),",","Joint Limits",sim1.model.jnt_range[j])

# print(sim1.model.nq)
print(sim1.model.nv)
# print(sim1.data.contact[0])

#Joint Limits: 
# Shoulder Pan: -2.285398 0.714602
# Shoulder Lift; -0.5236 1.3963
# Upper roll: -3.9 0.8
# Elbow flex: -2.3213 0
# Fore roll: 0 0
# Wrirst flex: -2.094 0
# Wrist roll: 0 0

#Desired Joint Positions and Joint Vel for the Feedback Control using PID
des_qvel = np.zeros(7)
# des_qpos = np.zeros(7)
# des_qpos = np.array([-0.655,-0.123,-0.8500,-0.567,0.25,0.5,0])

 # Current Default value on 27th Jan used in pr2_env
des_qpos = np.array([0.15,0.65,0,-1,0,-1,0])
# des_qpos = np.array([-0.6,0,-2,-1, 0,-1,0])
# Second joint to lift : .15, -0.4, -3, -1.05, 0 -1, 0
#  des_qpos = np.array([-0,-0.65,-4.5,-3,-1.570,-2.093,2])
# des_qpos_arr = np.zeros((sim_horizon,7))
des_qpos_arr = np.transpose([des_qpos]*sim_horizon)
des_qvel_arr = np.transpose([des_qvel]*sim_horizon)

for i in range(sim1.model.nbody):
    print("Body ID:",i,",","Body Name:",sim1.model.body_id2name(i))

for i in range(sim1.model.ngeom):
    print("Geom ID:",i,",","Geom Name:",sim1.model.geom_id2name(i))

#little-nem
# prop_name1 = "geom_friction"
# Using dof_frictionloss instead of adjusting the friction coefficients
# for now.
# dof - Degree of freedom, so I have to use all specific ids of the dofs 
# of the joints
prop_name2 = "dof_damping"
prop_name3 = "dof_frictionloss"
prop_all2 = getattr(sim1.model, prop_name2)
prop_all3 = getattr(sim1.model, prop_name3)
print(prop_all2)
print(prop_all3)
print(len(prop_all2))
print(len(prop_all3))





#NVidia Parameter estimation -- I guess for this we need to create a 
# xml for sure - we can't get away without this.

# model_xml = ET.parse(MODEL_PATH)
# model_root = model_xml.getroot()
# robot_joints = model_root.findall(".//worldbody//body[@name='robot']//joint")
# print("no. of robot joints:", len(robot_joints))
# print(robot_joints)

# dampings = [5,5,5,5,5,5,5,5,5,5,5,5]
# for robot_joint, damping in zip(robot_joints, dampings):
#     robot_joint.set('damping', '{:3f}'.format(damping))

# prop_name2 = "dof_damping"
# prop_all2 = getattr(sim1.model, prop_name2)
# print(prop_all2)


# def object_ids(obj_name):
#     obj_id = {}

#     try:
#         obj_id['body_id'] = sim1.model.body_name2id(obj_name)
#     except:
#         print("No Body found with given body name")
#         pass

    # try:
    #     obj_id['geom_id'] = sim1.model.geom_name2id(obj_name)
    # except:
    #     print("No geom found wih given Geom name")
    #     pass

    # try:
    #     obj_id['joint_id'] = sim1.model.joint_name2id(obj_name)
    # except:
    #     print("No joint found wih given joint name")
    #     pass

    # return obj_id

##################### Randomizing robot's body/link masses #######################
bod_names = [n for n in sim1.model.body_names if n.startswith('r_')]
# only randomizing certain robot links masses from shoulder pan to 
# wrist_roll
num_bod_r = len(bod_names)
# subtracting 5 gripper link indices which we are not randomizing
num_bod = num_bod_r - 5 
print(num_bod)
prop_name = "body_mass"
prop_all = getattr(sim1.model, prop_name)
print(prop_all)
print(len(prop_all))
new_body_mass = [5]*num_bod
print(new_body_mass)
body_id_arr = np.zeros(num_bod,dtype=np.int32)
for i in range(num_bod):
    body_id_arr[i] = sim1.model.body_name2id(bod_names[i])
print(body_id_arr)
cnt = 1
for j in range(len(prop_all)):
    # print("j:",j)
    if j==body_id_arr[cnt-1]:
        # print("cnt:",cnt)
        prop_all[j] = new_body_mass[cnt-1]
        cnt = cnt + 1
        if cnt > num_bod:
            break

# This method is useful for calculating all the other derived properties
# from the properties that we changed such as the body mass,shape,size, etc
# to avoid inconsisten physical simulation of the robot
#sim.set_constants()
sim1.set_constants()
print(prop_all)

################# To change the parameters values in mujoco_py ##########
prop_name2 = "dof_damping"
prop_name3 = "dof_frictionloss"
prop_all2 = getattr(sim1.model, prop_name2)
prop_all3 = getattr(sim1.model,prop_name3)
print(prop_all2)
# print(done_list)
# # print(info_list)
# # print(re_list)
print(prop_all3)


#sim1.model.joint_names gives us set of all joint names which we converted 
# to a list 
# defining a list of new damping and frictionloss coefficients for all the
# joints and their respective dofs
# so their size should be equal to number of dofs 
# new_dof_damping = [0,0,0,0,0,0,20,10,30,20,30,10,30,20,0.1,0.1,0.1,0.1]
# new_dof_frictionloss = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
#                         0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

new_dof_damping = [20,10,30,20,30,10,30,20,0.1,0.1,0.1,0.1]
new_dof_frictionloss = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
jnt_names = [n for n in sim1.model.joint_names]
print(jnt_names)
joint_dof_len = np.zeros(len(jnt_names))
prop_frictionloss = getattr(sim1.model,"dof_frictionloss")
prop_damping = getattr(sim1.model,"dof_damping")
length = 0
for i in range(len(jnt_names)):
    jnt_qvel_arr = sim1.data.get_joint_qvel(jnt_names[i])

    # The commented part of the code is useful if there is any object or puck
    # that is defined in the env
    # also range changes to len(jnt_ranges)-1 and the last two lines come
    # in the scope of else]if condition
    # if type(jnt_qvel_arr) != np.float64:
    #     length = len(jnt_qvel_arr)
    #     prop_frictionloss[i:i+length] = new_dof_frictionloss[i:i+length]
    #     prop_damping[i:i+length] = new_dof_damping[i:i+length]
    # elif type(jnt_qvel_arr) == np.float64:
    #     # length = 0

    prop_frictionloss[i+length-1] = new_dof_frictionloss[i+length-1]
    prop_damping[i+length-1] = new_dof_damping[i+length-1]




print(joint_dof_len)
print(prop_frictionloss)
print(prop_damping)

# new_damping = 1
# new_frictionloss = 0.5
# object_type2 = prop_name2.split('_')[0]
# object_type3 = prop_name3.split('_')[0]
# # object_type_id = object_type2 + '_id'
# prop_all2[list(upper_roll_id.values())[0]] = new_damping
# prop_all3[list(upper_roll_id.values())[0]] = new_frictionloss

prop_name2 = "dof_damping"
prop_name3 = "dof_frictionloss"
prop_all2 = getattr(sim1.model, prop_name2)
prop_all3 = getattr(sim1.model,prop_name3)
print(prop_all2)
print(prop_all3)

#Essential variables for the PID Controller
e_int = np.zeros((sim_horizon,7))
torque = np.zeros_like(e_int)
joint_pos = np.zeros((sim_horizon,7))
joint_vel = np.zeros((sim_horizon,7))
joint_ref = np.zeros_like(joint_vel)
bias_force = np.zeros_like(joint_vel)


#Most of the values from ROS wiki Github page  
#Test values
# kp_arr = np.array([1200,600,500,350,150,150,150])
# # kp_arr = np.array([600,300,250,175,75,75,75])
# kd_arr = np.array([25,20,12,9,12,12,9])
# # ki_arr = np.array([400,350,300,225,150,150,150])
# ki_arr = np.array([0,0,0,0,0,0,0])

#PD Controller with Ki = 0
#Values from ROS Wiki Healthcare robotics lab 
kp_arr = np.array([2400,1200,1000,700,300,300,300])
ki_arr = np.array([0,0,0,0,0,0,0])
kd_arr = np.array([18,10,6,4,6,4,4])


#Third Try: Really Good Results but starting offset really large
# kp_arr = np.array([140,170,300,100,300,200,200])
# ki_arr = np.array([0,0,0,0,0,0,0])
# kd_arr = np.array([30,30,7,5,20,8,8])

site_model = []
site_data = []
#Function Definition for PID Controller
def pid_controller(act_id,kp_dum,kd_dum,ki_dum,e,e_dot,e_int,bias_force):
    return (kp_dum[act_id]*e + kd_dum[act_id]*e_dot + ki_dum[act_id]*e_int) 

for i in range(sim_horizon):
    sim1.step()
    # print(len(sim1.data.qfrc_bias))
    for j in range(len(sim1.data.ctrl)):
        idx = sim1.model.jnt_qposadr[sim1.model.actuator_trnid[j,0]]
        joint_pos[i,j] = sim1.data.qpos[idx]
        joint_vel[i,j] = sim1.data.qvel[idx]
        bias_force[i,j] = sim1.data.qfrc_bias[idx]
        sim1.data.qfrc_applied[idx] = bias_force[i,j]
        if des_qpos[j] < sim1.model.jnt_range[sim1.model.actuator_trnid[j,0]][0]:
            des_qpos[j] = sim1.model.jnt_range[sim1.model.actuator_trnid[j,0]][0]
        elif des_qpos[j] > sim1.model.jnt_range[sim1.model.actuator_trnid[j,0]][1]:
            des_qpos[j] = sim1.model.jnt_range[sim1.model.actuator_trnid[j,0]][1]
        e = des_qpos[j] - joint_pos[i,j]
        e_dot = des_qvel[j] - joint_vel[i,j]
        e_int[i,j] = e_int[i-1,j] + e*dt
        pid_output= pid_controller(j,kp_arr,kd_arr,ki_arr,e,e_dot,e_int[i,j],bias_force[i,j])
        # sim1.data.ctrl[j] = pid_output
        
        if pid_output > sim1.model.actuator_ctrlrange[j][1]:
            sim1.data.ctrl[j] = sim1.model.actuator_ctrlrange[j][1]
        elif pid_output < sim1.model.actuator_ctrlrange[j][0]:
            sim1.data.ctrl[j] = sim1.model.actuator_ctrlrange[j][0]
        else:
            sim1.data.ctrl[j] = pid_output
        torque[i,j] = sim1.data.ctrl[j]
        joint_ref[i,j] = sim1.data.ctrl[j]
    
    # print("Site pos from model:", sim1.model.site_pos)
    # print("site pos from data:", sim1.data.site_xpos)
    site_model.append(sim1.model.site_pos)
    site_data.append(sim1.data.site_xpos)
    viewer1.render()

# print(sim1.data.contact)
# print(e_int[:2,:])
std_err = abs(joint_pos[sim_horizon-1,:]- des_qpos)
print("Steady State Error:",std_err)

site_frm_mdl = np.array(site_model)
site_frm_data = np.array(site_data)
print(site_frm_mdl.shape)

num_steps = np.arange(sim_horizon)
# plt.plot(num_steps,joint_pos[:,0],num_steps,joint_pos[:,1],num_steps,joint_pos[:,2],num_steps,joint_pos[:,3])
# fig,ax = plt.subplots(4,2)
# ax[0,0].plot(num_steps,joint_pos[:,0],num_steps,des_qpos_arr[0,:])
# ax[0,0].set_title('Shoulder Pan')
# ax[0,1].plot(num_steps,joint_pos[:,1],num_steps,des_qpos_arr[1,:])
# ax[0,1].set_title('Shoulder Lift')
# ax[1,0].plot(num_steps,joint_pos[:,2],num_steps,des_qpos_arr[2,:])
# ax[1,0].set_title('Upper arm roll')
# ax[1,1].plot(num_steps,joint_pos[:,3],num_steps,des_qpos_arr[3,:])
# ax[1,1].set_title('Elbow flex')
# ax[2,0].plot(num_steps,joint_pos[:,4],num_steps,des_qpos_arr[4,:])
# ax[2,0].set_title('Lower arm roll')
# ax[2,1].plot(num_steps,joint_pos[:,5],num_steps,des_qpos_arr[5,:])
# ax[2,1].set_title('Wrist flex')
# ax[3,0].plot(num_steps,joint_pos[:,6],num_steps,des_qpos_arr[6,:])
# ax[3,0].set_title('Wrist roll')


# for i in ax.flat:
#     i.set(xlabel='Time', ylabel='Radians')
# # for j in ax.flat:
# #     j.label_outer()
# # plt.legend(["joint angle","Reference"])
# # plt.legend(["Shoulder Pan","Shoulder Lift","Upper arm roll", "Elbow FLex"])
# plt.show()

# plt.figure()
# plt.plot(num_steps,site_frm_mdl[:,0,0],num_steps,site_frm_data[:,0,0])
# plt.legend(["site1-State1-model","site1-state1-data"])
# plt.show()
# plt.figure()
# plt.plot(num_steps,site_frm_mdl[:,0,1],num_steps,site_frm_data[:,0,1])
# plt.legend(["site1-State2-model","site1-state2-data"])
# plt.show()
# plt.figure()
# plt.plot(num_steps,site_frm_mdl[:,0,2],num_steps,site_frm_data[:,0,2])
# plt.legend(["site1-State3-model","site1-state3-data"])
# plt.show()
# plt.figure()
# plt.plot(num_steps,site_frm_mdl[:,1,0],num_steps,site_frm_data[:,1,0])
# plt.legend(["site2-State1-model","site2-state1-data"])
# plt.show()
# plt.figure()
# plt.plot(num_steps,site_frm_mdl[:,1,1],num_steps,site_frm_data[:,1,1])
# plt.legend(["site2-State2-model","site2-state2-data"])
# plt.show()
# plt.figure()
# plt.plot(num_steps,site_frm_mdl[:,1,2],num_steps,site_frm_data[:,1,2])
# plt.legend(["site2-State3-model","site2-state3-data"])
# plt.show()
# plt.figure()
# plt.plot(num_steps,torque[:,0],num_steps,torque[:,1])
# plt.xlabel("Time")
# plt.ylabel("Torque")
# plt.legend(["Shoulder Pan Joint","Shoulder Lift Joint"])
# plt.show()

# plt.figure()
# plt.plot(num_steps,torque[:,2],num_steps,torque[:,3])
# plt.xlabel("Time")
# plt.ylabel("Torque")
# plt.legend(["Upper Arm Roll","Elbow Flex"])
# plt.show()

# plt.figure()
# plt.plot(num_steps,bias_force[:,0],num_steps,bias_force[:,1],num_steps,bias_force[:,2],num_steps,bias_force[:,3])
# plt.xlabel("Time")
# plt.ylabel("Bias and Gravity")
# plt.legend(["Shoulder Pan","Shoulder Lift","Upper arm roll", "Elbow FLex"])
# plt.show()


## The PR2 actually has a spring counter balance for offsetting the gravity
# So there are two appraoches:
# 1. Find out the gravity bias acting on the shoulder lift joint each time
# step and add/subtract it.
# the gravity bias is subtracted from the "forward dynamics equation" so to 
# compensate it by including it as an applied force that gets summed up with 
# the actuation forces resulting in a "forward acceleration"
#  http://mujoco.org/book/computation.html#General

# 2. Increase the ctrl range for the motor on the Shoulder lift joint

#Puck has 6 DOF joint(free joint) and all the other defined joints have only 
#hinge joints -- which are 26 in number making nv(degrees of freedom=32), now 
#arresting the base joints with 3DOFs.. resulting in total of 29 DOF .
# idx = sim1.model.jnt_qposadr[sim1.model.actuator_trnid[j,0]] to find the index 
# of the joint for which the actuator is dedicated to. so that joint vel and joint
# bias forces can be captured
