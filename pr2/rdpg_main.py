import os
import numpy as np 
import random
import gym
import pr2_push
from random_environment import RandomizeEnvironment
from ddpg_agent import DDPGAgent
from memory_buffer import EpisodicBuffer, ReplayBuffer
import matplotlib.pyplot as plt


directory = "ddpg_rdpg_50"

if not os.path.exists(directory):
    os.makedirs(directory)

# Constants in python are written in Capitals
BATCH_SIZE = 32
EPISODES = 4000
EPISODE_LENGTH = 500
MAX_BUFFER_SIZE = 2000
GAMMA = 0.99
TAU = 1e-3    #default value - 1e-3
ACTOR_LR = 5e-4
CRITIC_LR = 5e-4
K = 0.8 #Probability of the HER sampling as given in paper
WARMUP_TIME = 100000 

env_to_randomize = "PR2Reach-v0"
dyn_par_ranges = np.array([[0,2],[0,1],[5,10]])
randomized_env = RandomizeEnvironment(env_to_randomize,dyn_par_ranges)

#Manually coded for now - Dimensions of parameters that are randomized 
dim_dyn_par = 31
base_env = gym.make(env_to_randomize)

agent = DDPGAgent(base_env, dim_dyn_par, GAMMA, TAU, ACTOR_LR, CRITIC_LR)

replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
count = 0
rewards = []
her_rewards = []
avg_rewards = 0
###
#### AS MENTIONED BELOW THIS IS FIRST TYPE OF IMPLEMENTATION
# # # ###
for ep in range(EPISODES):
    # print(ep)
    randomized_env.sample()
    env,dim_dyn_par,env_params = randomized_env.env_n_parameters()
    # print("Environment Parameters:",env_params)
    obs_dict = env.reset()
    goal = obs_dict['desired_goal']
    # print("Goal from main",goal)
    episode = EpisodicBuffer(goal,env_params,EPISODE_LENGTH)
    episode_reward = 0
    for ep_step in range(EPISODE_LENGTH):
        # env.render()

        obs = obs_dict['observation']
        # print(obs)
        goal = obs_dict['desired_goal']
        # print("goal episode start",goal)
        action_noise = agent.get_action_noise()
        action_frm_policy = agent.action_input_frm_network(obs,goal)
        # print("policy action shape", action_frm_policy.shape)
        action  = action_frm_policy.reshape(7,) + action_noise
        # print("final action shape", action.shape)
        # action = action_frm_policy.reshape(7,)

        new_obs_dict,reward,done,info = env.step(action)
        episode_reward += reward
        new_obs = new_obs_dict['observation']
        achieved_g = new_obs_dict['achieved_goal']
        # print(achieved_g)
        episode.add_episode_step(obs,action,reward,new_obs,achieved_g,done)
        obs_dict = new_obs_dict

    replay_buffer.add_episode(episode)
    rewards.append(episode_reward)
    # print("Episodic Buffer",episode)
    # print("Normal Replay buffer",replay_buffer)   
    ###Implementing HER with Final State as the Desired Goal 
    # for Goal Sampling   
    her_episode_reward = 0 
    if random.random() < K:
        # print("HER Resampled")
        her_goal = obs_dict['achieved_goal']
        # print("HER Achieved goal", her_goal)
        replay = EpisodicBuffer(her_goal, env_params,EPISODE_LENGTH)
        # print("HER Sampled_goal", her_goal)
        # print("HER Environment paramteres", env_params)
        for ep_step in episode.episode:
            obs, action, reward, new_obs, achieved_g, done = ep_step
            #compute reward expects three args (third arg: info)
            new_reward = env.compute_reward(achieved_g,her_goal,0)
            replay.add_episode_step(obs,action,new_reward,new_obs,achieved_g,done)
            her_episode_reward += new_reward
        replay_buffer.add_episode(replay)
        # print("Replayed EPisodicbuffer",replay)
    # print("Replay buffer after HER",replay_buffer)
    her_rewards.append(her_episode_reward)
    if replay_buffer.len_buffer() > BATCH_SIZE:
        count += 1 
        print("Number of Optimization Steps", count)
        # print("replay_buffer current length",replay_buffer.len_buffer())
        agent.policy_update(replay_buffer, BATCH_SIZE,EPISODE_LENGTH)


randomized_env.close_env()

agent.save_model(directory)
