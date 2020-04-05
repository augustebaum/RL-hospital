from hospital import *
from learning import *
import matplotlib.pyplot as plt


# Doctors
# One of level 0 that has probability of being done of 0.2
# One of level 0 that has probability of being done of 0.1
# One of level 1 that has probability of being done of 0.1
# One of level 2 that has probability of being done of 0.05
doctors = [Doctor(0, 0.2),
           Doctor(1, 0.1)]
#types = [2, 1, 1]
#probs = [0.2, 0.1, 0.1, 0.05]

# Hospital with occupancy of 20 people 
# Patient of type 0 five times as likely than of type 2
# Patient of type 1 twice as likely than of type 2
hospital = Hospital(20, doctors, [1, 1000])

#hospital.simulate(hospital.policy_random, limit = 30)

##############################################################################
# the variables needed for the Sarsa algorithm below and the result - currently 
# not a useful result :(
num_episodes = 1
num_steps = 60
gamma = 1
alpha = 1 / num_steps
epsilon = 0.1

#hospital.queues[0] = [Patient(0, 1), Patient(2,3),Patient(1, 5), Patient(0, 5), Patient(2, 5), Patient(2, 5)] 
#hospital.pretty_print()
#hospital.next_step(1)
#hospital.pretty_print()

t_list, Q_weights, total_reward_per_episode, timeline_episodes = sarsa(hospital, feature_1, gamma, alpha, epsilon, num_episodes, num_steps)
# 
print("\nQ_weights:\n", Q_weights)
print("\nThe list with termination episodes:\n", t_list)
print("\nTotal number of terminated episode: ", len(t_list))
print("\nrewards per episode:\n", total_reward_per_episode)
# #plt.figure(1)
# #plt.plot(timeline, total_reward_per_step)
plt.figure(2)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Sarsa")
plt.plot(timeline_episodes, total_reward_per_episode)
plt.plot(timeline_episodes, np.zeros(num_episodes))


