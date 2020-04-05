from hospital import *
from learning import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    t_list, Q_weights, total_reward_per_episode, timeline_episodes = simulation1(featurisation = feature_1, num_episodes = 100, num_steps = 300)
    print("Q_weights:", Q_weights)
    print("The list with termination episodes:", t_list)
    print("Total number of terminated episodes:", len(t_list))
    print("Rewards per episode:", total_reward_per_episode)

    rewards_curve(total_reward_per_episode, len(total_reward_per_episode))
    # plt.figure(1)
    # plt.plot(timeline, total_reward_per_step)
    # plt.figure(2)

    plt.show()

def simulation1(featurisation = feature_1, num_episodes = 30, num_steps = 40, gamma = 0.85, alpha = None, epsilon = 0.1):
    """
    Two doctors, one of type 0 and one of type 1, and only patients of type 1.
    The expected result is that all patients are dispatched to queue 1.
    """
    if alpha is None: alpha = 1/num_steps

    # One of level 0 that has probability of being done of 0.2
    # One of level 1 that has probability of being done of 0.1
    doctors = [Doctor(0, 0.2), Doctor(1, 0.1)]
    
    # Hospital with occupancy of 20 people 
    # Patient of type 1 5000 times more likely than type 0
    hospital = Hospital(20, doctors, [1, 5000])

    return sarsa(hospital, featurisation, gamma, alpha, epsilon, num_episodes, num_steps)

def rewards_curve(rewards, num_episodes):
    fig, ax = plt.subplots()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    # plt.title("Simulation 1")
    ax.plot(range(num_episodes), rewards)
    ax.plot(range(num_episodes), np.zeros(num_episodes))

if __name__ == '__main__':
    main()
