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
run_naive_policy      -> if a patient is of type X then they are assigned to a doctor of type X
"""

run_each_featurisation_once_and_plot = True

run_feature_1  = False
run_feature_7  = False
run_feature_8  = False
run_feature_9  = False
run_feature_10 = False
run_feature_11 = False
run_feature_12 = False
run_feature_13 = False
number_of_experiments = 1

run_naive_policy = False

calcRewards = False



##############################################################################
##############################################################################
##############################################################################

# these are the relative probabilites of each patient arriving - the index in 
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

def test_17(
    algorithm,
    capacity_hospital,
    number_steps,
    number_episodes,
    p_arr_prob,
    doctors,
    feature,
    rand_rewards = None,
    p_prob_test = None,
    gamma = 0.9,
    alpha = None,
    epsilon = 0.1,
    plot_type = "hist",
    title1 = "",
    title2 = "",
    earlyRewards = True,
    capacity_penalty = False,
    reward_evolution = False,
    naive_rewards = None):
    """
    Inputs
    ---------
    algorithm         - sarsa and Q-learning available
    capacity_hospital - Maximum number of people that could be waiting in all the queues together.
    number_steps      - Number of steps in an episode that does not terminate earlier than it is supposed to.
    number_episodes   - Number of episodes
    p_arr_prob        - Relative probabilities for different patients arriving.
                        The index in the list corresponds to patient's type.
                        [1, 1, 4] means that patient of type 0 has a probability of 1/6 to arrive,
                        while patient of type 2 has a probability of 4/6 to arrive at each step.
    doctors           - The doctors currently in the hospital. 
                        Doctor(x, y) means a doctor of type x with a probability of y to cure a patient on any step.
    feature           - Featurisation of the state.
    rand_rewards      - Rewards obtained with a random policy
    p_prob_test       - Relative probabilities for patient arrivals used during testing
                        (by default, the same as training probabilities)
    gamma             - Geometric discount factor for rewards
    alpha             - Learning rate
    epsilon           - Probability of choosing action randomly
    plot_type         - Histogram ("hist") and heat map ("heat") available, or anything else for no plot
    title1, title2    - Title for the plots
    earlyRewards      - True means the rewards are allocated directly when the patient is sent to a specific queue.
                        Otherwise rewards are recognized when the patient reaches the doctor.
    capacity_penalty  - True means when the capacity is reached not only the episode terminates
                        but there is also a penalty for letting the hospital get full.
    reward_evolution  - True means that the function will also plot the rewards for each episode
    naive_rewards     - True means that the reward_evolution plot will also include the
                        episodic rewards achieved using the naive rewards.

    Output
    ---------
    Produces 2 plots - one for the best allocation of patients
        and one for the rewards evolution during the learning process.
    Some extra information for the leaning process is also printed out.
    """

    # an instance of the Hospital object (defined in hospital.py)
    hospital = Hospital(capacity_hospital, doctors, p_arr_prob)
    
    # function for the sarsa algorithm.
    # Q_weights - the weight matrix, 1 weight vector for each action in the simulation
    # total_reward_per_episode - a list with the reward for each episode
    # t_list - List of info for when each episode is terminated (if terminated)
    # --- if alpha = None then alpha is calculated as 1 / number_steps
    t_list, Q_weights, total_reward_per_episode = algorithm(
            hospital,
            feature,
            gamma,
            alpha,
            epsilon,
            number_episodes,
            number_steps,
            earlyRewards,
            capacity_penalty)
    
    
    # Simulate the learned policy.
    # props - a matrix for the relative distribution of patients in the queues.
    # rewards - a list with the reward acquired at each step in the simulation
    # cured - total number of cured patients during the simulation
    # time - total time waited by cured patients
    # cured_types - cured patients by types
    
    # If you want to use different patient arrival probabilities for testing, a new hospital is created
    if p_prob_test is not None:
        hospital = Hospital(capacity_hospital, doctors, p_prob_test)

    if not(title1):
        title1 = ("Early" if earlyRewards else "Late")+\
                 " rewards and "+\
                 ("a" if capacity_penalty else "no")+\
                 " capacity penalty"

    props, rewards, cured, time, cured_types = simulate(
            hospital,
            feature,
            Q_weights,
            steps = number_steps,
            plot = plot_type,
            title = title1,
            checkBefore = earlyRewards,
            cap_penalty = capacity_penalty)
    
    # Below we get reward results for a completely random action taking , i.e. epsilon = 1
    # t_list_r, Q_optimal_weights_r, total_reward_per_episode_r = algorithm(hospital, feature, 0, 0, 1, number_episodes, number_steps)
    
    
    # A plot that shows the episodic reward evolution during the learning phase
    # this is also informative of how fast the algorithm is learning
    if reward_evolution:
        rewards_curve(total_reward_per_episode,
                      number_episodes,
                      title2,
                      naive_rewards)
    
    
    # Extra information to be printed for the first figure
    print_extra_info(rewards, cured, number_steps, cured_types, sum(map(sum, time)), title1)
    
    return props, rewards, cured, time, cured_types , total_reward_per_episode

# simple func to write to a txt file
def writeToFile(fileLocation, rewardList):
    myfile = open(fileLocation,"a")
    for reward in rewardList:
        myfile.write(str(reward) + "\n")
    myfile.close()

    
def misalloc(alloc_matrix):
    return np.sum(np.tril(alloc_matrix, -1))



# sets the currently needed parameters in the "test" function 
# to reduce the code
def feature_experiment(num_exp, featurisation, fileLocRew = None, fileLocMis = None, fileLocEvol = None, plot = None, rew_graph = False, naive_rew = None, tit1 = "", tit2 = ""):
    
    exp_rewards = []
    misalloc_list = []
    rewards_per_ep = []
    
    for experiment in range(num_exp):
    
        props, rewards, c, t, cr, rewperep = test_17(algorithm = sarsa,
                                                     capacity_hospital = capacity,
                                                     number_steps = number_of_steps,
                                                     number_episodes = number_of_episodes,
                                                     p_arr_prob = p_arr_prob,
                                                     doctors = doctors_1,
                                                     feature = featurisation,
                                                     rand_rewards = None,
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
        
        if len(rewards_per_ep) == 0:
            rewards_per_ep = np.array(rewperep)
        else:
            rewards_per_ep += np.array(rewperep)
    
        
        exp_rewards.append(sum(rewards))
        
        misallocations = misalloc(props)
        misalloc_list.append(misallocations)
        
    rewards_per_ep = rewards_per_ep / num_exp  
    
    # save all the reward info in a file
    if fileLocRew != None:
        writeToFile(fileLocRew, exp_rewards)
    # save the info about misallocations
    if fileLocMis != None:
        writeToFile(fileLocMis, misalloc_list)
    # saves rewards evolution information
    if fileLocEvol != None:
        writeToFile(fileLocEvol, rewards_per_ep)
    
    if plot == None:
        print("\n\nThe table results are below: ")
        print("The list with the rewards", exp_rewards)
        print("The average reward after {} experiments is {}".format(num_exp, np.mean(exp_rewards)))
        print("The median is {}".format(np.median(np.array(exp_rewards))))
        print("The standard deviation is {}".format(np.std(np.array(exp_rewards))))
    

    
    

if run_each_featurisation_once_and_plot:
    
    #feature_experiment(1, feature_1, plot = "hist", rew_graph = True, 
    #                   tit1 = "feature_1 patient allocation", tit2 = "feature_1 rewards during learning")
    #feature_experiment(1, feature_7, plot = "hist", rew_graph = True,
    #                   tit1 = "feature_7 patient allocation", tit2 = "feature_7 rewards during learning")
    feature_experiment(1, feature_12, plot = "hist", rew_graph = True,
                       tit1 = "feature_12 patient allocation", tit2 = "feature_12 rewards during learning")


elif run_feature_1:
    feature_experiment(number_of_experiments,feature_1,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature1.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature1.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature1.txt",
                       tit1 = "feature 1 current experiment")


elif run_feature_7:
    feature_experiment(number_of_experiments, feature_7,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature7.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature7.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature7.txt",
                       tit1 = "feature 7 current experiment")


elif run_feature_8:
    feature_experiment(number_of_experiments,feature_8,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature8.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature8.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature8.txt",
                       tit1 = "feature 8 current experiment")


elif run_feature_9:
    feature_experiment(number_of_experiments,feature_9,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature9.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature9.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature9.txt",
                       tit1 = "feature 9 current experiment")
    

elif run_feature_10:
    feature_experiment(number_of_experiments,feature_10,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature10.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature10.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature10.txt",
                       tit1 = "feature 10 current experiment")
    

elif run_feature_11:
    feature_experiment(number_of_experiments,feature_11,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature11.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature11.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature11.txt",
                       tit1 = "feature 11 current experiment")
    
    
elif run_feature_12:
    feature_experiment(number_of_experiments, feature_12,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature12.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature12.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature12.txt",
                       tit1 = "feature 12 current experiment")


elif run_feature_13:
    feature_experiment(number_of_experiments, feature_13,
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature13.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\misallocFeature13.txt",
                       r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolFeature13.txt",
                       tit1 = "feature 13 current experiment")
    
    
elif run_naive_policy:
    reward_list = []
    reward_evol = np.zeros(100)
    
    
    # r_naives is accumulated rewards on each step
    for i in range(1):
        evol_temp = []
        
        for experiment in range(number_of_experiments):
            #if experiment == 0:
            plot = False
            p_naive, r_naives, *rest = simulate(hospital_r, naive = True, steps = number_of_steps, plot = plot)
                
            reward_list.append(sum(r_naives))
            evol_temp.append(sum(r_naives))
            
        reward_evol += np.array(evol_temp)
    reward_evol = reward_evol / 50
                
           # else:
            #    plot = False
             #   p_naive, r_naives, *rest = simulate(hospital_r, naive = True, steps = number_of_steps, plot = plot)
              #  
               # reward_list.append(sum(r_naives)) # needs some fixing
                #
                #reward_evol.append(sum(r_naives))
        
    writeToFile(r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\evolNaive.txt", reward_evol)
    #writeToFile(r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsNaive.txt", reward_list)
    print("The list with the rewards", reward_list)
    print("The average reward after {} experiments is {}".format(number_of_experiments, np.mean(reward_list)))
    print("The median is {}".format(np.median(np.array(reward_list))))
    print("The standard deviation is {}".format(np.std(np.array(reward_list))))
    print(reward_evol)

# temporary here
if calcRewards:
    my_list = []
    read = open(r"C:\Users\todor\Documents\UCL_MATH_ECON\Year_3\Foundations_ML\RL_project\rewardsFeature12.txt","r")
    for line in read:
        my_list.append(float(line.rstrip()))
    read.close()
    print("The length of the list: {}\n".format(len(my_list)))
    print("The median: {}\n".format(np.median(np.array(my_list))))
    print("The standard deviation : {}".format(np.std(np.array(my_list))))











