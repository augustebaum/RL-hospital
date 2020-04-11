from hospital import *
from learning import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # holds the average reward after n simulations
    av_r_sim = 0
    # holds the average after n random action simulations
    av_r = 0
    # n is the number of cycles (i.e. simulations run)
    n = 1
    for _ in range(n):
        (t_list, Q_weights, total_reward_per_episode), max_rewards = simulation1()
        
        # results from random action below, used only for the plot -> nothing else printed
        (t_list_r, Q_weights_r, total_reward_per_episode_r), max_rewards = simulation1(epsilon = 1)
        print("Q_weights:", Q_weights)
        print("The list with termination episodes:", t_list)
        print("Total number of terminated episodes:", len(t_list))
        print("Rewards per episode:", total_reward_per_episode)
        
        max_r_list = np.zeros(len(total_reward_per_episode))
        max_r_list.fill(max_rewards)
    
        rewards_curve(max_r_list, total_reward_per_episode_r, total_reward_per_episode, len(total_reward_per_episode))
        # plt.figure(1)
        # plt.plot(timeline, total_reward_per_step)
        # plt.figure(2)
    
        plt.show()
        print("Average episodic reward is: {}".format(np.average(total_reward_per_episode)))
        av_r_sim += np.average(total_reward_per_episode)
        av_r += np.average(total_reward_per_episode_r)
    print("\nThe average reward after {} simulations is {}".format(n, av_r_sim/n))
    print("\nThe average reward after {} random simulations is {}".format(n, av_r/n))

def simulation1(featurisation = feature_1, num_episodes = 300, num_steps = 100, gamma = 0.85, alpha = None, epsilon = 0.1):
    """
    Two doctors, one of type 0 and one of type 1, and only patients of type 1.
    The expected result is that all patients are dispatched to queue 1.
    """
    if alpha is None: alpha = 1/num_steps

    # One of level 0 that has probability of being done of 0.2
    # One of level 1 that has probability of being done of 0.1
    doctors = [Doctor(0, 0.1),
           Doctor(1, 0.1),
           Doctor(2, 0.1),
           Doctor(3, 0.1),
           Doctor(4, 0.1),
           Doctor(5, 0.1),
           Doctor(6, 0.6)]
    hospital = Hospital(20, doctors, [1, 1, 1, 1, 1, 1, 1])
    
    # Hospital with occupancy of 20 people 
    # Patient of type 1 5000 times more likely than type 0
    # the list holds to the relative probabilities of patients occurring 
    # the index of the list's element corresponds to the patient's type
    
    max_reward = - (hospital.max_average_reward() * num_steps)

    return sarsa(hospital, featurisation, gamma, alpha, epsilon, num_episodes, num_steps), max_reward

def rewards_curve(max_rewards, rand_rewards, sim_rewards, num_episodes):
    fig, ax = plt.subplots()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Simulation 1")
    ax.plot(range(num_episodes), sim_rewards, "-b", label = "simulation rewards")
    ax.plot(range(num_episodes), max_rewards, "-g", label = "max reward line")
    ax.plot(range(num_episodes), rand_rewards, "-r", label = "random actions")
    ax.legend()

if __name__ == '__main__':
    main()
