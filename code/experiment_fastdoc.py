from hospital import *
from learning import *
from experiments_extra_functions import *
import numpy as np
import matplotlib.pyplot as plt
""" 
The current experiment will focus on a hospital object with 4 different types 
of doctors and varying probability of each type of patient to arrive. 

The goal here is to understand how the agent's behaviour changes as model parameters are changed.
Out of the 4 doctors, doctor 3 (the most highly skilled) also happens to be highly efficient compared to the others (treats patients faster).
This means that it could be advantageous for the agent to give a large proportion of patients to this doctor, even if they have low priority.
However, if there are many high priority patients (who cannot be treated by anyone else), giving doctor 3 many low priority patients means increasing the waiting time of priority 3 patients, which is more costly that for low priority patients.
Hence, the agent should start allocating to lower priority patients.
Specifically, the first trial is with equally likely arrivals
"""
feature = feature_7 # One-hot
capacity_hospital = 100
number_steps = 200
number_episodes = 100


##############################################

# print("\nThe average step reward after the simulation with naive policy is:", np.mean(r_naives))

# p is the probability that the arriving patient is of type 3
p_array = np.linspace(0.1, 0.9, 10)
# Number of people in queue 3
queue3 = []
# Number of people in queue 3 that are of type 3
queue3type3 = []

for p in p_array:
    arrival_rates = [1, 1, 1, round(3*p/(1-0.99999*p))]

    p_slow = 0.4
    p_fast = 0.8
    doctors_1 = [Doctor(0, p_slow),
                 Doctor(1, p_slow),
                 Doctor(2, p_slow),
                 Doctor(3, p_fast)]


    # this hospital object used only to calculate the random rewards list
    hospital_r = Hospital( capacity_hospital, doctors_1, [1, 1, 1, 1])

    # Random policy (total_reward_per_episode_r is needed in `test`)
    t_list_r, Q_optimal_weights_r, total_reward_per_episode_r = sarsa( hospital_r, feature, 0, 0, 1, number_episodes, number_steps)

    # Run hospital with the naive policy for number_steps steps
    # Record allocations and plot heatmap
    # p_naive, r_naives = simulate_naive( hospital_r, steps = number_steps, plot = "map")

    # Train, simulate and gather:
    # - The number of non-type 3 patients in queue 3
    # - The number of time that 3-patients waited?

    # Testing using equal arrival probabilities to provide unbiased account
    props, rewards, cured, time, cured_types =\
        test(
            sarsa,
            capacity_hospital,
            number_steps,
            number_episodes,
            arrival_rates,
            doctors_1,
            feature,
            total_reward_per_episode_r,
            p_prob_test = [1, 1, 1, 1],
            gamma = 0.9,
            alpha = None,
            epsilon = 0.1,
            plot_type = None,
            title1 = "Type 3 patients arrive {:.0%} of the time during training".format(p),
            title2 = "Reward evolution for the picture above")

    print("Proportions:\n", props)
    # props[3,3]
    queue3.append(sum(props[:,3]))
    queue3type3.append(props[3,3])

print(queue3)
print(queue3type3)
plt.figure()
plt.plot(p_array, queue3, label="# of patients in queue 3")
plt.plot(p_array, queue3type3, label="# of patients of type 3 in queue 3")
plt.legend()

plt.show()

