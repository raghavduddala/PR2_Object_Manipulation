import numpy as np


# Hardcoded values for PID Gains for each joint
kp_arr = np.array([2400,1200,1000,700,300,300,300])
ki_arr = np.array([0,0,0,0,0,0,0])
kd_arr = np.array([18,10,6,4,6,4,4])

#Somehow have to use the sub steps offered in gym to do the position 
# control correctly and the joint velocities must come to zero
# before the next target joint pose is given.
#Hardcoded this 
des_qvel = np.array([0,0,0,0,0,0,0])
joint_pos = np.zeros(7)
joint_vel = np.zeros(7)

def robot_obs(sim):
    """
    Returns Joint positions and Joint velocities
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('r_')]
        return(
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    
    return np.zeros(0),np.zeros(0)




#PID Controller 
def pid_controller(act_id,kp_dum,kd_dum,ki_dum,e,e_dot):
    return (kp_dum[act_id]*e + kd_dum[act_id]*e_dot) 


def set_joint_action(sim,action):
    """
    Method to Control the PR2 arm with the given action from NN
    Uses "pid_controller" and "set_action_limit" methods

    Using a custom PD Controller
    Action from NN is 
    Args:

    idx - gives the index of the joint that the actuator is defined for
    """
    if sim.data.ctrl is not None:
        assert action.shape[0] == 7 
        for i in range(len(sim.data.ctrl)):
            idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i,0]]
            joint_pos[i] = sim.data.qpos[idx]
            joint_vel[i] = sim.data.qvel[idx] 
            sim.data.qfrc_applied[idx] = sim.data.qfrc_bias[idx]
            #Checking Joint Limits
            if action[i] < sim.model.jnt_range[sim.model.actuator_trnid[i,0]][0]:
                action[i] = sim.model.jnt_range[sim.model.actuator_trnid[i,0]][0]
            elif action[i] > sim.model.jnt_range[sim.model.actuator_trnid[i,0]][1]:
                action[i] = sim.model.jnt_range[sim.model.actuator_trnid[i,0]][1]
            e = action[i] - joint_pos[i]
            e_dot = des_qvel[i]- joint_vel[i]
            pid_output = pid_controller(i,kp_arr,ki_arr,kd_arr,e,e_dot)
            #Checking Controller Torque Limits
            if pid_output < sim.model.actuator_ctrlrange[i][0]:
                sim.data.ctrl[i] = sim.model.actuator_ctrlrange[i][0]
            elif pid_output > sim.model.actuator_ctrlrange[i][1]:
                sim.data.ctrl[i] = sim.model.actuator_ctrlrange[i][1]
            else:
                sim.data.ctrl[i] = pid_output

