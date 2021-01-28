from gym.envs.registration import register

# register(
#     id="PR2-v0",
#     entry_point="pr2_push.env:PR2Env",
#     max_episode_steps = "",
# )

# for reward_type in ['sparse','dense']:
#     suffix = 'Dense' if reward_type =='dense' else ''
#     kwargs = {
#     'reward_type' : reward_type,
#     }
    
# register(
#     id = "PR2Reach-v0".format(suffix),
#     entry_point = "pr2_push.env:PR2ReachEnv",
#     kwargs=kwargs,
#     max_episode_steps = 50,
#     )
    
# register(
#     id = "PR2Push-v0".format(suffix),
#     entry_point = "pr2_push.env:PR2PushEnv",
#     max_episode_steps = 100,
#     )



# PR2 Environment registration
# entry point in the module pr2_push from PR2Env Class
# max episode_steps has not been implemented 


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch
    register(
        id='PR2Reach{}-v0'.format(suffix),
        entry_point= 'pr2_push.env:PR2ReachEnv',
        kwargs=kwargs,
        max_episode_steps=200,
    )

    register(
        id='PR2Push{}-v0'.format(suffix),
        entry_point='pr2_push.env:PR2PushEnv',
        kwargs=kwargs,
        max_episode_steps=200,
    )