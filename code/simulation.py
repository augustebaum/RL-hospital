from hospital import *
from Sarsa import *
import matplotlib.pyplot as plt


# Doctors
# One of level 0 that has probability of being done of 0.2
# One of level 0 that has probability of being done of 0.1
# One of level 1 that has probability of being done of 0.1
# One of level 2 that has probability of being done of 0.05
doctors = [Doctor(0, 0.2),
           Doctor(0, 0.1),
           Doctor(1, 0.1),
           Doctor(2, 0.05)]
#types = [2, 1, 1]
probs = [0.2, 0.1, 0.1, 0.05]

# Hospital with occupancy of 20 people 
# Patient of type 0 five times as likely than of type 2
# Patient of type 1 twice as likely than of type 2
hospital = Hospital(20, doctors, [5, 2, 1])

#hospital.simulate(hospital.policy_random, limit = 30)

##############################################################################
# the variables needed for the Sarsa algorithm below and the result - currently 
# not a useful result :(
num_episodes = 80
num_steps = 40
gamma = 0.85
alpha = 1 / num_steps
epsilon = 0.1

Q_weights, total_reward_per_episode, timeline_episodes = sarsa(hospital, gamma, alpha, epsilon, num_episodes, num_steps)

print("\nQ_weights:\n", Q_weights)
#plt.figure(1)
#plt.plot(timeline, total_reward_per_step)
plt.figure(2)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.plot(timeline_episodes, total_reward_per_episode)


