This will be a custom mujoco gym environment for accepting custom built xmls 
1. First xml will be of Cartpole 
2. PR2 Robot xml from Vikash Kumar



Seems like I can follow the path and the directories as the original mujoco gym environments. This Implies,
I need not regster a custom mujoco environment( base class environment) but instead register only the custom inverted pendulum environment and use the custom mujoco environment just as a super class. Let us check the implementation by running a cartpole simulation all from the custom environments

If this works, I can delete the cartpole-random folder or else it can be used to test the dynamic randomizations
and 
later work on the PR2 robot xml and the PR2 gym Env.


For now, let the path be as follows:

```
custom-mujoco-gym
|
|__custom_mujoco_gym
|  |__custom_envs
|  |  |____init__.py
|  |  |__custom_cartpole_env.py
|  |  |__custom_mujoco_gym_env.py             ---- Super class for the mujoco custom cartpole environment
|  |  
|  |__Robotic_envs (for PR2-- Under Construction)
|  |
|  |____init__.py
|
|__README.md
|__setup.py

```