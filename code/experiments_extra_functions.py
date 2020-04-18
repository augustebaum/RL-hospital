from hospital import *
from simulation import *
from learning import *
import numpy as np
import matplotlib.pyplot as plt

def print_extra_info(rewards, cured, number_steps, cured_types, time, info):
    """
    Prints out some extra metrics about the process of learning.
    """
    print(40*"#")
    print("Extra data for {}\n\n".format(info))
    print("\nThe average step reward after the simulation with the fixed Q_weights is : {}"
      .format(np.mean(rewards)))
    print("\n{} patients were cured during the simulation of {} steps.\n".format(cured, number_steps))
    print("Patients cured by types: \n{}\n".format(cured_types))
    print("Total time waited by the cured patients: {}\n".format(time))

def test(algorithm, capacity_hospital, number_steps, number_episodes, p_arr_prob, doctors,
         feature, rand_rewards, gamma = 0.9, alpha = None, epsilon = 0.1,
         plot_type = "hist", title1 = "title 1 unknown", title2 = "title 2 unknown",
         earlyRewards = True, capacity_penalty = False):
    """
    Inputs
    ---------
    algorithm - sarsa and Q-learning available
    capacity_hospital - capacity of the hospital, which is the max number of 
                        people that could be waiting in all the queues together.
    number_steps - number of steps in an episode that does not terminate earlier
                   than it is supposed to.
    number_episodes - number of episodes
    p_arr_prob - a list of the relative probabilities for different patients arriving.
                The index in the list corresponds to patient's type. [1, 1, 4] 
                means that patient of type 0 has a probability of 1/6 to arrive,
                while patient of type 2 has a probabilityof 4/6 to arrive at each step.
    doctors - the doctors currently in the hospital. Doctor(x, y) means a doctor
              of type x with a probability of y to cure a patient on any step.
    feature - the currently used featurisation of the state space. The definition of each 
              featurisation is in learning.py
    rand_rewards - list of rewards following a random action taking
    gamma - geometric discount factor for rewards
    alpha - step size
    epsilon - variable for how greedy the policy is
    plot_type - histogram("hist") and heat map available("heat")
    title1, title2 - title for the plots
    earlyRewards - True means the rewards are allocated directly when the patient 
                   is sent to a specific queue. Otherwise rewards are recognized 
                   when the patient reaches the doctor.
    capacity_penalty - if True then when the capacity is reached not only the episode
                       terminates but there is also a negative reward acquired.
    Output
    ---------
    this function produces 2 plots - one for the best allocation of patients
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
    
    
    # a function to simulate the learned policy.
    # props - a matrix for the relative distribution of patients in the queues.
    # rewards - a list with the reward acquired at each step in the simulation
    # cured - total number of cured patients during the simulation
    # time - total time waited by cured patients
    # cured_types - cured patients by types
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
    # This result is independent on
   # t_list_r, Q_optimal_weights_r, total_reward_per_episode_r = sarsa(hospital, feature, 0, 0, 1, number_episodes, number_steps)
    
    
    # A plot that shows the episodic reward evolution during the learning phase
    # this is also informative of how fast the algorithm is learning
    rewards_curve(0, rand_rewards, total_reward_per_episode, number_episodes, title2)
    
    
    
    # Extra information to be printed for the first figure
    print_extra_info(rewards, cured, number_steps, cured_types, time, title1)
    
    