from gym.envs.registration import register

register(
    id = "Custom_inverted_pendulum-v1",
    entry_point = "custom_mujoco_gym.custom_envs:CustomInvertedPendulumEnv",
    max_episode_steps = 1000, 
    )


    # The initial default maximum episode steps is 1000, can be changed
    # Another parameter can be reward threshold 