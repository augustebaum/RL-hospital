from hospital import *
# from simulation import *
from learning import *
from experiments_extra_functions import *
import numpy as np
import matplotlib.pyplot as plt
""" 
The current experiment will focus on a hospital object with 6 different types 
of doctors and equal probability of each type of patient to arrive. 

Currently the main arguments that are changed between each test are the algorithm,
the featurisation and whether the rewards are allocated earlier or later.

The available algorithms are sarsa and Q-learning.
There are several available featurisations, most notably the difference is that 
some are encoded as one-hot vectors.
Rewards are recognized either immediately when a patient is assigned to a doctor's
queue or later - when the patient reaches the doctor after having waited in the queue.

"""
#############################################
# common arguments in this box for now
feature = feature_7
capacity_hospital = 100
number_steps = 100
number_episodes = 100

# these are the realtive probabilites of each patient arriving - the index in 
# the list corresponds to the patient's type. If we had a list [1,2,4] then this would mean
# that patients of type 0 have a probability of 1/7 to visit the hospital on 
# any given step, patients of type 1 - probability of 2/7 and type 2 - 4/7
p_arr_prob = [1, 1, 1, 1, 1, 1]
p_arr_prob_2 = [1, 2, 3, 4, 5, 6]

# doctors_1 is currently used for all tests (it looks like a good choice
doctors_1 = [Doctor(0, 0.1),
             Doctor(1, 0.1),
             Doctor(2, 0.9),
             Doctor(3, 0.1),
             Doctor(4, 0.5),
             Doctor(5, 0.1)]

doctors_2 = [Doctor(0, 0.1),
             Doctor(1, 0.1),
             Doctor(2, 0.1),
             Doctor(3, 0.1),
             Doctor(4,   1),
             Doctor(5, 0.1)]
# this hospital object used only to calculate the random rewards list
hospital_r = Hospital(capacity_hospital, doctors_1, p_arr_prob)
t_list_r, Q_optimal_weights_r, total_reward_per_episode_r = sarsa(hospital_r, feature, 0, 0, 1, number_episodes, number_steps)
##############################################

# Run hospital with the naive policy for number_steps steps
# Record allocations and plot heatmap
p_naive, r_naives = simulate_naive(hospital_r, steps = number_steps, plot = "hist")
print("\nThe average step reward after the simulation with naive policy is:",np.mean(r_naives))



# the title show clearly what it is
# each test produces 2 figures

# First Test (SARSA Algorithm)
test(sarsa,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(1.1)Sarsa + earlyReward + Capacity_penalty",
    title2 = "(1.1)Reward evolution for the picture above",
    earlyRewards = True,capacity_penalty = True)
 
test(sarsa,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(1.2)Sarsa + early rewards",
    title2 = "(1.2)Reward evolution for the picture above",
    earlyRewards = True,capacity_penalty = False)


test(sarsa,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(1.3)Sarsa + capacity_penalty",
    title2 = "(1.3)Reward evolution for the picture above",
    earlyRewards = False,capacity_penalty = True)

test(sarsa,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(1.4)Sarsa + no penalty",
    title2 = "(1.4)Reward evolution for the picture above",
    earlyRewards = False,capacity_penalty = False)




# Second Test (QL Algorithm)
test(ql,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(2.1)Q-learnin + early rewards + capacity_penalty",
    title2 = "(2.1)Reward evolution for the picture above",
    earlyRewards = True, capacity_penalty = True) 


test(ql,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(2.2)Q-learning + early rewards",
    title2 = "(2.2)Reward evolution for the picture above",
    earlyRewards = True,capacity_penalty = False)

test(ql,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(2.3)Q-learning + capacity_penalty",
    title2 = "(2.3)Reward evolution for the picture above",
    earlyRewards = False,capacity_penalty =True)

test(ql,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors_1,
    feature,
    total_reward_per_episode_r,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "(2.4)Q-learning + no penalty",
    title2 = "(2.4)Reward evolution for the picture above",
    earlyRewards = False,capacity_penalty =False)



