import os
import numpy as np 
import random
import gym
import pr2_push
from random_environment import RandomizeEnvironment
from ddpg_agent import DDPGAgent
from memory_buffer import EpisodicBuffer, ReplayBuffer



directory = "ddpg_checkpoints"

if not os.path.exists(directory):
    os.makedirs(directory)

# seed = 1234

"""
In the Sim2real transfer paper, they sample the mass, damping, friction 
logarithmetically which can be done using np.logspace
but for now, we are using only Uniform distribution 
Mass ranges from 0 to 2
Friction loss changes from 0 to 1
Damping changing changes from 5 to 10
"""

#################################################################
### TESTING THE RANDOM ENVIRONMENT - Update - Working without any errors
#################################################################

# env_to_randomize = "PR2Reach-v0"
# dyn_par_ranges = np.array([[0,2],[0,1],[5,10]])
# randomized_env = RandomizeEnvironment(env_to_randomize,dyn_par_ranges)
# randomized_env.sample()

# env,dim_dyn_par,env_params = randomized_env.env_n_parameters()
# # print("Dimensions of parameters randomized:",dim_dyn_par)
# # print(env.action_space.shape[0])
# # print(env.observation_space['desired_goal'].shape[0])
# print("Desired goal before reset", env.observation_space['desired_goal'])
# # print("Observation in main before reset", env.observation_space['observation'])
# obs_first = env.reset()
# print("Desired goal after reset", env.observation_space['desired_goal'])
# # # print(obs_first['observation'])
# # print("Observation in main after reset", obs_first['observation'])
# print("Parameters that are randomized:", env_params)
# # print(env_params.shape)
# re_list = []
# info_list = []
# done_list = []
# for _ in range(1000):
#     env.render()
#     obs, reward, done, info = env.step(env.action_space.sample())
#     re_list.append(reward)
#     info_list.append(info)
#     done_list.append(done)
# env.close()

# print(done_list)
# print(info_list)
# print(re_list)


#################################################################
### TRAINING THE MODEL
#Have yet to defien the max_episode_steps in init for env
#################################################################
"""
25-D State space 
11 Joint Positions (Joint Angles)
11 Joint Velocities
03 Gripper Site Pos (X,Y,Z) Coordinations

7-D Action space
 Joint angle Offsets

HER Impementation based on :
Hindsight Experience Replay
https://arxiv.org/pdf/1707.01495.pdf
USsing only one goal strategy i.e using Final State achieved at the end
of the episode as the achieved goal in HER

Version 18th January changing to 5*e-3 for "tau"
will change to take random actions with a warm up time buffer
"""

################## HYPERPARAMETERS
BATCH_SIZE = 32
EPISODES = 3000
EPISODE_LENGTH = 50
MAX_BUFFER_SIZE = 2000
GAMMA = 0.99
TAU = 5e-3    #default value - 1e-3
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
k = 0.8 #Probability of the HER sampling as given in paper
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
# ###
# #### AS MENTIONED BELOW THIS IS FIRST TYPE OF IMPLEMENTATION
# # ###
for ep in range(EPISODES):
    # print(ep)
    randomized_env.sample()
    env,dim_dyn_par,env_params = randomized_env.env_n_parameters()
    # print("Environment Parameters:",env_params)
    obs_dict = env.reset()
    goal = obs_dict['desired_goal']
    episode = EpisodicBuffer(goal,env_params,EPISODE_LENGTH)

    for ep_step in range(EPISODE_LENGTH):
        # env.render()

        obs = obs_dict['observation']
        goal = obs_dict['desired_goal']
        action_noise = agent.get_action_noise()
        action_frm_policy = agent.action_input_frm_network(obs,goal)
        # print("policy action shape", action_frm_policy.shape)
        action  = action_frm_policy.reshape(7,) + action_noise
        # print("final action shape", action.shape)
        # action = action_frm_policy.reshape(7,)

        new_obs_dict,reward,done,info = env.step(action)
        new_obs = new_obs_dict['observation']
        achieved_g = new_obs_dict['achieved_goal']

        episode.add_episode_step(obs,action,reward,new_obs,achieved_g,done)
        obs_dict = new_obs_dict

    replay_buffer.add_episode(episode)
    # print("Episodic Buffer",episode)
    # print("Normal Replay buffer",replay_buffer)   
    ###Implementing HER with Final State as the Desired Goal 
    # for Goal Sampling    
    if random.random() < k:
        # print("HER Resampled")
        her_goal = obs_dict['achieved_goal']
        replay = EpisodicBuffer(her_goal, env_params,EPISODE_LENGTH)
        for ep_step in episode.episode:
            obs, action, reward, new_obs, achieved_g, done = ep_step
            #compute reward expects three args (third arg: info)
            new_reward = env.compute_reward(achieved_g,her_goal,0)
            replay.add_episode_step(obs,action,new_reward,new_obs,achieved_g,done)
        
        replay_buffer.add_episode(replay)
        # print("Replayed EPisodicbuffer",replay)
    # print("Replay buffer after HER",replay_buffer)
    if replay_buffer.len_buffer() > BATCH_SIZE:
        count += 1 
        print("Number of Optimization Steps", count)
        # print("replay_buffer current length",replay_buffer.len_buffer())
        agent.policy_update(replay_buffer, BATCH_SIZE,EPISODE_LENGTH)


randomized_env.close_env()

agent.save_model(directory)
 
 #########################################################################
 ### TESTING THE MODEL 
 #########################################################################
# agent.load_model(directory)
# agent.eval_model()
# randomized_env.sample()
# info_list = []
# done_list = []
# re_list = []
# env,dim_dyn_par,env_params = randomized_env.env_n_parameters()
# obs_dict = env.reset()
# for _ in range(EPISODE_LENGTH):
#     env.render()
#     obs = obs_dict['observation']
#     goal = obs_dict['desired_goal']
#     action_frm_policy = agent.action_input_frm_network(obs,goal)
#     action = action_frm_policy.reshape(7,)
#     obs, reward, done, info = env.step(action)
#     re_list.append(reward)
#     info_list.append(info)
#     done_list.append(done)
# env.close()

# print(done_list)
# print(re_list)

###In the paper, the success rate is determined as the portion of episodes
#where goal is fulfilled at the end of the episode
# i.e the policy is evaluated over 100 episodes
#info is a dict used to store whenever the end-effector is almost at the 
# desired goal position
# info['is_success'] is_success is the key in this dict which is 
# actually a function whose value is 1 or 0


successful_episode = 0
agent.load_model(directory)
agent.eval_model()

suc_list = []
action_list = []
action_list_list = []
reward_list = []
reward_list_list = []
############### Hyperparameters
TEST_EPISODES = 100

for ep in range(TEST_EPISODES):
    randomized_env.sample()
    env,dim_dyn_par,env_params = randomized_env.env_n_parameters()

    obs_dict = env.reset()
    goal = obs_dict['desired_goal']
    episode = EpisodicBuffer(goal,env_params,EPISODE_LENGTH)
    for ep_step in range(EPISODE_LENGTH):
        # env.render()

        obs = obs_dict['observation']
        goal = obs_dict['desired_goal']
        # action_noise = agent.get_action_noise()
        action_frm_policy = agent.action_input_frm_network(obs,goal)
        # action  = action_frm_policy + action_noise
        action = action_frm_policy.reshape(7,)
        # action_list.append(action)
        new_obs_dict,reward,done,info = env.step(action)
        # print("action from network", action)
        # print(info)
        reward_list.append(reward)
        new_obs = new_obs_dict['observation']
        achieved_g = new_obs_dict['achieved_goal']

        episode.add_episode_step(obs,action,reward,new_obs,achieved_g,done)
        obs_dict = new_obs_dict
    
    # action_list_list.append(action_list)
    suc = info['is_success']
    reward_list_list.append(reward_list)
    suc_list.append(suc)
    if info['is_success'] == 1.0:
        successful_episode += 1

randomized_env.close_env()

print(successful_episode)
print(suc_list)
# print(action_list_list)
# print(reward_list_list)
print("successrate for 100 episodes:",successful_episode/TEST_EPISODES)


# # #DDPG takes only states and goal 





#####One main mistake in the implementation is not storing the newly sampled
# goal in th buffer and storing only the rewards##### - Tomorrow morning
# Implementation work

#also can implement suitable RDPG implemntation from p-chris's implementation
#seems somewhat correct
"""
Have to fix the dimensions of the Environment Parameters that I access first
and use it to send as a input to the Critic NN.


1. Will have to do two versions of the same code, first method as
little-nem as also given in the paper(Paper - Batch contains 128 Episodes 
with each episode consisting 100 steps)

2. A different implementation where the current episode is accessed and
changed according to HER probability 
Pseudocode:
for ep in Num_Episodes:
    for i=0:N-1 in episode:
        Store transitions directly in replay buffer R
    if HER_prob <= 0.8:
         New_goal = last_state
         for i = 0: N-1 in episode:
             Replay the transitions
             Update the "New Reward" with the "New Goal'
             Store transitions
    if R>batch_size:
        call policy_update         
"""

