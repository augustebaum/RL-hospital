from hospital import *
from learning import *
import numpy as np
from matplotlib import rc
# rc('text', usetex=True)
import matplotlib.pyplot as plt

def main():
    n = 2
    # holds the average reward of each simulation
    av_r_sim = np.zeros(n)
    # holds the average of each random action simulations
    av_r = np.zeros(n)
    # n is the number of cycles (i.e. simulations run)
    plt.figure()
    for i in range(n):
        (t_list, Q_weights, total_reward_per_episode), max_rewards = simulation1(num_episodes = 20)
        
        # results from random action below, used only for the plot -> nothing else printed
        (t_list_r, Q_weights_r, total_reward_per_episode_r), max_rewards = simulation1(featurisation = feature_5, epsilon = 1, num_episodes = 20)
        print("Simulation no. ", i+1)
        print("Q_weights:\n", Q_weights)
        print("The list with termination episodes:\n", t_list)
        print("Total number of terminated episodes:", len(t_list))
        print("Rewards per episode:", total_reward_per_episode)
        max_r_list = np.zeros(len(total_reward_per_episode))
        max_r_list.fill(max_rewards)
    
        plt.subplot(1, n+1, i+1)
        rewards_curve(max_r_list, total_reward_per_episode_r, total_reward_per_episode, len(total_reward_per_episode), i = i, legend = (i==0))
    
        print("Average episodic reward is: {}".format(np.average(total_reward_per_episode)))
        av_r_sim[i] = np.average(total_reward_per_episode)
        av_r[i] = np.average(total_reward_per_episode_r)
    if n > 1:
        plt.subplot(1, n+1, n+1)
        plt.plot(range(n), av_r_sim, range(n), av_r)
    plt.show()
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

def rewards_curve(max_rewards, rand_rewards, sim_rewards, num_episodes, i = None, legend = True):
    # if fig is None:
    #     fig, ax = plt.subplots(tight_layout = True)
    # else:
    #     ax = fig.subplots()
    plt.title("Simulation "+str(i) if i else "Simulation 1")
    # ax.plot(range(num_episodes), sim_rewards, "-b", figure = fig, label = "Learned policy")
    # ax.plot(range(num_episodes), max_rewards, "-g", figure = fig, label = "$r=0$ line")
    # ax.plot(range(num_episodes), rand_rewards, "-r", figure = fig, label = "Random policy")
    plt.plot(range(num_episodes), sim_rewards, "-b", label = "Learned policy")
    plt.plot(range(num_episodes), max_rewards, "-g", label = "$r=0$ line")
    plt.plot(range(num_episodes), rand_rewards, "-r", label = "Random policy")
    if legend:
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

    # return ax

if __name__ == '__main__':
    main()
