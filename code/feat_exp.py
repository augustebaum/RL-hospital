from hospital import *
from learning import *
# from experiments_extra_functions import *
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
##############################################################################
##############################################################################
"""
Control panel for those who are currently grading this project.
In this box there are options that will easily print out certain results
regarding this experiment.
Set only one of the variables below to "True" at a time.

run_each_featurisation_once_and_plot -> for each featurisation there will be 2 plots
             1st plot will show the allocation of patient after a simulation of the learned policy
             2nd will show the rewards at each episode during the learning process
number_of_experiments -> defines how many times the learning process is repeated for a certain feature
                      -> Our data about mean rewards and standard deviation is gathered after 
                         a significant number of experiments (100 for each feature) so that it is accurate.
                      -> Note that that the runtime for 100 experiments on a certain feature might take up to 5 minutes
number_of_experiments -> initially set to 10  -  Data in report histogram taken after 100 for each feature
"""

run_each_featurisation_once_and_plot = True
run_feature_1 = False
run_feature_7 = False
run_feature_12 = False

number_of_experiments = 10


##############################################################################
##############################################################################
##############################################################################

# these are the realtive probabilites of each patient arriving - the index in 
# the list corresponds to the patient's type. If we had a list [1,2,4] then this would mean
# that patients of type 0 have a probability of 1/7 to visit the hospital on 
# any given step, patients of type 1 - probability of 2/7 and type 2 - 4/7
p_arr_prob = [1, 1, 1, 1, 1, 1]


# doctors_1 is currently used for all tests (it looks like a good choice
doctors_1 = [Doctor(0, 0.1),
             Doctor(1, 0.1),
             Doctor(2, 0.9),
             Doctor(3, 0.4),
             Doctor(4, 0.1),
             Doctor(5, 0.5)]
# max number of people that could be waiting in the hospital
capacity = 100
number_of_steps = 100
number_of_episodes = 100

# this hospital object used only to calculate the random rewards list
hospital_r = Hospital(capacity, doctors_1, p_arr_prob)
# t_list_r, Q_optimal_weights_r, total_reward_per_episode_r = sarsa(hospital_r, feature_7, 0, 0, 1, number_of_episodes, number_of_steps)

# Run hospital with the naive policy for number_steps steps
# Record allocations and plot heatmap
p_naive, r_naives, *rest = simulate(hospital_r, naive = True, steps = number_of_steps, plot = "hist", title = "Naive policy patient allocation")
print("\nThe total reward after the simulation with naive policy:", sum(r_naives))

# sets the currently needed parameters in the "test" function 
# to reduce the code
def feature_experiment(num_exp, featurisation, plot = None, rew_graph = False, naive_rew = None, tit1 = "", tit2 = ""):
    
    exp_rewards = []
    
    for experiment in range(num_exp):
    
        p, rewards, c, t, cr = test(algorithm = sarsa,
                                    capacity_hospital = capacity,
                                    number_steps = number_of_steps,
                                    number_episodes = number_of_episodes,
                                    p_arr_prob = p_arr_prob,
                                    doctors = doctors_1,
                                    feature = featurisation,
                                    rand_rewards = r_naives,
                                    gamma = 0.9,
                                    alpha = None,
                                    epsilon = 0.1,
                                    plot_type = plot,
                                    title1 = tit1,
                                    title2 = tit2,
                                    earlyRewards = True,
                                    capacity_penalty = False,
                                    reward_evolution = rew_graph,
                                    naive_rewards = naive_rew) 
        
        exp_rewards.append(sum(rewards))
    
    if plot == None:
        print("\n\nThe table results are below: ")
        print("The list with the rewards", exp_rewards)
        print("The average reward after {} experiments is {}".format(num_exp, np.mean(exp_rewards)))
        print("The median is {}".format(np.median(np.array(exp_rewards))))
        print("The standard deviation is {}".format(np.std(np.array(exp_rewards))))
    
    
    

if run_each_featurisation_once_and_plot:
    
    feature_experiment(1, feature_1, plot = "hist", rew_graph = True, naive_rew = r_naives, 
                       tit1 = "feature_1 patient allocation", tit2 = "feature_1 rewards during learning")
    feature_experiment(1, feature_7, plot = "hist", rew_graph = True, naive_rew = r_naives,
                       tit1 = "feature_7 patient allocation", tit2 = "feature_7 rewards during learning")
    feature_experiment(1, feature_12, plot = "hist", rew_graph = True, naive_rew = r_naives,
                       tit1 = "feature_12 patient allocation", tit2 = "feature_12 rewards during learning")


elif run_feature_1:
    feature_experiment(number_of_experiments, feature_1, tit1 = "feature 1 current experiment")


elif run_feature_7:
    feature_experiment(number_of_experiments, feature_7, tit1 = "feature 7 current experiment")


elif run_feature_12:
    feature_experiment(number_of_experiments, feature_12, tit1 = "feature 12 current experiment")





















