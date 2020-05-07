from hospital import *
from learning import *
from fastdoc_exp import *
import numpy as np
import matplotlib.pyplot as plt

"""
You can run this file as is:
``` python3 feat_exp.py ```
which will generate data for experiment 1 and plot it.
Depending on the training parameter below (Number_of_experiments),
the plots can be more or less accurate.

Note: the exact plots in the report are created after a lot of data was saved in files.
The exact plots will require Number_of_experiments = 300. However, this
will take more than 20 minutes. 

The parameter is currently set to 15, which should take ~3 minutes to run. 
This should produce reasonable plots.

"""

##############################################################################
Number_of_experiments = 15                                                   #
##############################################################################

# these are the relative probabilites of each patient arriving - the index in 
# the list corresponds to the patient's type. If we had a list [1,2,4] then this would mean
# that patients of type 0 have a probability of 1/7 to visit the hospital on 
# any given step, patients of type 1 - probability of 2/7 and type 2 - 4/7
p_arr_prob = [1, 1, 1, 1, 1, 1]

# a list with the features to be used
features = [feature_1, feature_7, feature_12, feature_13]
featNames = {0:"feature_1", 1:"feature_7", 2:"feature12", 3:"feature_13"}

# list of doctors; Doctor(type, 0 <= efficiency <= 1)
doctors = [Doctor(0, 0.1),
             Doctor(1, 0.1),
             Doctor(2, 0.9),
             Doctor(3, 0.4),
             Doctor(4, 0.1),
             Doctor(5, 0.5)]

# max number of people that could be waiting in the hospital
capacity = 100
number_of_steps = 100
number_of_episodes = 100

# this hospital object used only to calculate data for the naive policy
hospital_object = Hospital(capacity, doctors, p_arr_prob)


# used to create a matrix that will hold the necessary data; i.e. number of rows
n_features = 4

# matrices to hold the reward data needed for the final plots
rew_sim_matrix = [[] for _ in range(n_features)]
rew_learning_matrix = np.zeros((n_features, number_of_episodes))


def main(num_exp):
    
    # produce data for the naive policy
    naive_median_quantiles, naive_evol = naive_policy()
    
    # produce data for all the features
    for j, feature in enumerate(features):
        for i in range(num_exp):
            
            # plot only once and for feature_12 only
            if i == 0 and j == 2:
                plot = "hist"
            else:
                plot = False
            
            # learning and simulation function
            learning_rew, sim_rew, *_  = test_exp(sarsa,
                                                  capacity,
                                                  number_of_steps,
                                                  number_of_episodes,
                                                  p_arr_prob, 
                                                  doctors,
                                                  feature,
                                                  plot_type = plot,
                                                  title1 = featNames[j])
            
            # add data to the matrix with simulation rewards
            simulation_rewards(sum(sim_rew), j)
            
            # add data to the matrix with learning rewards evolution
            learning_rewards(learning_rew, j, num_exp)
            
    # output results such as Figure 3 - reward evolution during learning       
    show_learning_curves(rew_learning_matrix, naive_evol)
    
    # calculate the data needed for Figure 2
    calc_median_quantiles(rew_sim_matrix)
    
    # output results such as Figure 2 - rewards after simulations
    show_simulation_rewards(rew_sim_matrix, naive_median_quantiles)
    
    
    
def calc_median_quantiles(rew_matrix):
    """
    Calculates the median reward and the distance
    from median to 20th and 80th percentile, for error bars
    
    Inputs
    ----------
    rew_matrix - each row corresponds to the episodic rewrard of a feature
    """
    for i in range(len(rew_matrix)):
        rew_matrix[i] = errors(rew_matrix[i])
    
def simulation_rewards(rew, index):
    """
    Adds reward from an episode to a feature's reward list
    
    Inputs
    ----------
    rew - current episodic reward for a specific feature
    index - index that corresponds to a feature
    """
    rew_sim_matrix[index].append(rew)

def learning_rewards(rew_per_ep, index, num_exp):
    """
    Updates the matrix with the reward evolution data (learning phase)
    
    Inputs
    ----------
    rew_per_ep - list with episodic rewards
    index - index that corresponds to one of the features
    num_exp - Number_of_Experiments
    """
    rew_learning_matrix[index] += np.array(rew_per_ep) / num_exp
    
def show_learning_curves(rew_matrix, naive_rew):
    """
    Plots the reward evolution (from the learning process)
    
    Inputs
    ----------
    rew_matrix - matrix with reward evolution lists for the features
    naive_rew - list with rewards from the naive policy
    """
    plt.plot(range(len(rew_matrix[0])), rew_matrix[0], "-b", label = "Feature 1")
    plt.plot(range(len(rew_matrix[1])), rew_matrix[1], "-g", label = "Feature 7")
    plt.plot(range(len(rew_matrix[2])), rew_matrix[2], "-r", label = "Feature 12")
    plt.plot(range(len(rew_matrix[3])), rew_matrix[3], "-c", label = "Feature 13")
    plt.plot(range(len(naive_rew)), naive_rew, "-m", label = "Naive policy")
    plt.xlabel("Episode")
    plt.ylabel("Reward (learning phase)")
    plt.legend()
    
def show_simulation_rewards(rew_matrix, naive_median_quantiles):
    """
    Plots the simulation (after learning) rewards for each feature (bar plot)
    
    Inputs
    ----------
    rew_matrix - holds the median and error bars for each feature
    naive_median_quantiles - holds the median and error bars for naive policy
    """
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    features = ['Naive policy', 'feat1', 'feat7', 'feat13', 'feat12']
    rewards = [naive_median_quantiles[0], rew_matrix[0][0], rew_matrix[1][0], rew_matrix[3][0], rew_matrix[2][0]]
    quantiles = [[naive_median_quantiles[1][0], rew_matrix[0][1][0], rew_matrix[1][1][0], rew_matrix[3][1][0], rew_matrix[2][1][0] ],
                 [naive_median_quantiles[1][1], rew_matrix[0][1][1], rew_matrix[1][1][1], rew_matrix[3][1][1], rew_matrix[2][1][1] ]]
    ax.bar(features, rewards, yerr = quantiles)
    plt.ylabel("Simulation rewards")
    plt.show()

def naive_policy():
    """
    naive policy - assign patient x to queue x
    There is no actual learning process for the naive policy.
    This return both a list with rewards and error bars (20th and 80th percentile)
    """
    print("Experiment 1 initiated")
    
    reward_list = [0 for _ in range(number_of_episodes)]
    
    for experiment in range(Number_of_experiments):
        for ep in range(number_of_episodes):
            
            p_naive, r_naives, *rest = simulate(hospital_object, naive = True, steps = number_of_steps)
            reward_list[ep] += (sum(r_naives)) / Number_of_experiments
                
    return errors(reward_list), reward_list
            
def test_exp(
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
    total_reward_per_episode - episodic rewards during the learning process
    rewards     - a list with rewards after a simulation with learned Qweights
    props       - a matrix that shows patient allocation in queues
    cured       - the number of patients seen by a doctor
    time        - total time waited by treated patients
    cured_types - dict with treated patients by types
    """

    # an instance of the Hospital object (defined in hospital.py)
    hospital = Hospital(capacity_hospital, doctors, p_arr_prob)
    
    # learning process with the currently used algorithm.
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
    
    
    # If you want to use different patient arrival probabilities for testing, a new hospital is created
    if p_prob_test is not None:
        hospital = Hospital(capacity_hospital, doctors, p_prob_test)

    if not(title1):
        title1 = ("Early" if earlyRewards else "Late")+\
                 " rewards and "+\
                 ("a" if capacity_penalty else "no")+\
                 " capacity penalty"
                 
    # Simulate the learned policy.
    # props - a matrix for the relative distribution of patients in the queues.
    # rewards - a list with the reward acquired at each step in the simulation
    # cured - total number of cured patients during the simulation
    # time - total time waited by cured patients
    # cured_types - cured patients by types    

    props, rewards, cured, time, cured_types, size = simulate(
            hospital,
            feature,
            Q_weights,
            steps = number_steps,
            plot = plot_type,
            title = title1,
            checkBefore = earlyRewards,
            cap_penalty = capacity_penalty)
    
    
    # A plot that shows the episodic reward evolution during the learning phase
    # this is also informative of how fast the algorithm is learning
    if reward_evolution:
        rewards_curve(total_reward_per_episode,
                      number_episodes,
                      title2,
                      naive_rewards)
    
    
    # Extra information to be printed for each simulation
    print_extra_info(rewards, cured, number_steps, cured_types, sum(map(sum, time)), title1)
    
    return total_reward_per_episode, rewards, props, cured, time, cured_types, size
    

if __name__ == "__main__":
    main(Number_of_experiments)
    
    