
"""
Update - January 27th, Found that the same action is given by the DDPG 
at every timestep during inference, It can be due to the fact that the
manipulability can be limited at those singular configurations i.e why 
Horizontal Joint velocities are not being achieved when the goal is 
sampled left and right 


The below guys also faced a similar issue of getting into Unstable Joint
Configurations

Learning to Run challenge solutions: Adapting reinforcement
learning methods for neuromusculoskeletal environments
https://arxiv.org/pdf/1804.00361.pdf

Will have to change the goal sampling bias so has to avoid Unstable 
joint configurations.

Approach:
    Should check joint torques being applied, if they are maxing out 
    and also the joint velocities
    Keeping the configuration of the first few joints to avoid the problem
    Setting a goal bias such that the goal is sampled away from the low
    manipulable joint configurations/change the initial joint configuration similar to that of fetch robot to avoid this issue and the goal will also be sampled like that
    Should check if the same problem persists in the RDPG 
    Can definitely choose an ensemble method/ or choose RSVG method(which gives
    stochastic actor output or should try PPO)

Changes:
    Re-scaling the inputs by a fcator of 1/20 in pr2_env similar to fetch_env to avoid issues on manipulability

Observations:
    The torques for the first few joints are maxing out, may be 
    manipulation is difficult 
"""