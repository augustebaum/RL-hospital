from hospital import *
from learning import *
from experiments_extra_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
# For exporting variables to file
from datetime import datetime
""" 
The current experiment will focus on a hospital object with 4 different types 
of doctors and varying probability of each type of patient to arrive. 
The goal here is to understand how the agent's behaviour changes as model parameters are changed.
Out of the 4 doctors, doctor 3 (the most highly skilled) also happens to be highly efficient compared to the others (treats patients faster).
This means that it could be advantageous for the agent to give a large proportion of patients to this doctor, even if they have low priority.
However, if there are many high priority patients (who cannot be treated by anyone else), giving doctor 3 many low priority patients increases the waiting time of priority 3 patients, which is more costly that for low priority patients.
Hence, the agent should allocate to more low skill doctors.
"""
feature = feature_7 # One-hot
capacity_hospital = 100
number_steps = 100
number_episodes = 100

##############################################

# Efficiency of low skill doctors
p_slow = 0.4
# Efficiency of high skill doctor
p_fast = 0.8

doctors_1 = [Doctor(0, p_slow),
             Doctor(1, p_slow),
             Doctor(2, p_slow),
             Doctor(3, p_fast)]

# p is the probability that the arriving patient is of type 3
p_array = np.linspace(0.1, 0.9, 5)
# Number of simulations for each p (to get error bars)
number_tries = 5
# Number of people in queue 3 for each p
queue3 = np.empty([number_tries, len(p_array)])
# Number of people in queue 3 that are of type 3 for each p
queue3type3 = np.empty([number_tries, len(p_array)])

for j, p in enumerate(p_array):
    arrival_rates = [1, 1, 1, round(3*p/(1-0.99999*p))]

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

    for i in range(number_tries):
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
        # queue3.append(sum(props[:,3]))
        # queue3type3.append(props[3,3])
        queue3[i, j] = sum(props[:,3])
        queue3type3[i, j] = props[3,3]

print(queue3)
print(queue3type3)

np.savez("exp2_data_"+datetime.now().strftime("%d-%m,%H:%M"), queue3, queue3type3)

av_q3 = np.mean(queue3, axis=0)
av_q3t3 = np.mean(queue3type3, axis=0)

yerr_q3 = 2*np.std(queue3, axis=0)
yerr_q3t3 = 2*np.std(queue3type3, axis=0)

print("2 std devs for queue3", yerr_q3)
print("2 std devs for queue3type3", yerr_q3t3)

plt.figure()
# plt.plot(p_array, queue3, label="Proportion of patients in queue 3")
plt.errorbar(p_array, av_q3, yerr=yerr_q3, elinewidth=1, capsize=2, label="Proportion of patients in queue 3")
# plt.plot(p_array, queue3type3, label="Proportion of patients of type 3 in queue 3")
plt.errorbar(p_array, av_q3t3, yerr=yerr_q3t3, elinewidth=1, capsize=2, label="Proportion of patients of type 3 in queue 3")
plt.legend()
plt.xlabel("Probability that arriving patient has type 3 during training")

plt.show()

