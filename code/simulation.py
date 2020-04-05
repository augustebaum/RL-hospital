from hospital import *
from learning import *
import matplotlib.pyplot as plt

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

    # Q_weights, total_reward_per_episode, timeline_episodes = sarsa(hospital, feature_1, gamma, alpha, epsilon, num_episodes, num_steps)
    return sarsa(hospital, featurisation, gamma, alpha, epsilon, num_episodes, num_steps)

def main():
    print("hello")
    t_list, Q_weights, total_reward_per_episode, timeline_episodes = simulation1(featurisation = feature_2, num_episodes = 50, num_steps = 100)
    print("\nQ_weights:\n", Q_weights)
    print("\nThe list with termination episodes:\n", t_list)
    print("\nTotal number of terminated episode: ", len(t_list))
    print("\nrewards per episode:\n", total_reward_per_episode)
    #plt.figure(1)
    #plt.plot(timeline, total_reward_per_step)
    # plt.figure(2)
    fig, ax = plt.subplots()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Simulation 1")
    print(total_reward_per_episode)
    plt.plot(timeline_episodes, total_reward_per_episode)
    plt.plot(timeline_episodes, np.zeros(num_episodes))

    plt.show()

if __name__ == '__main__':
    main()
