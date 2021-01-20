import gym 
import custom_mujoco_gym
import mujoco_py
import numpy as np 

# from mujoco_py import MjSim, load_model_from_path, MjViewer

# model = load_model_from_path("gym/gym/envs/robotics/assets/fetch/pick_and_place.xml")

# sim = MjSim(model)

# print(sim.data.ctrl)
# print(sim.model.actuator_trnid)
# print(sim.data.mocap_pos)
# print(sim.data.mocap_quat)


# env = gym.make('HandManipulateBlock-v0')
env = gym.make("FetchPush-v1")
# env = gym.make("FetchReach-v1")

# env = gym.make('Custom_inverted_pendulum-v1')
# print(sim.data.ctrl)
env.reset()
obs_list = []
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
    obs_list.append(obs)
env.close()

# print(len(obs_list))
# obs_arr = np.array(obs_list)
# # print(obs_arr)
# print(obs_arr.shape)

###inverted pendulum (cartpole) environment is already existing in gym
#should first understand it and then try to implement changes 
#and  changing the action space and randomizing the environment

# 