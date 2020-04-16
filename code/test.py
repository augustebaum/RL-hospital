from hospital import *
from simulation import *
from learning import *
import numpy as np
from matplotlib import rc
# rc('text', usetex=True)
import matplotlib.pyplot as plt

# Two doctors, one of type 0 and one of type 1
# Both have probability 0.1 of being done treating a patient
# Feel free to change these
#                     |
#                     v
doctors = [Doctor(0, 0.1),
           Doctor(1, 0.1)]
# The hospital can hold 20000 people -> the episode won't terminate early
# Each type of patient is equally likely to arrive (as many type 0 as type 1)
# Feel free to change these
#                                    |  |
#                                    v  v
hospital = Hospital(20000, doctors, [1, 1])

# Featurisation with only one-hot vectors
feature = feature_7
# Each episode ends after number_steps steps
number_steps = 100
# Learning ends after number_episodes episodes
number_episodes = 100

# Train learner with Q-learning
t_list, Q_weights, total_reward_per_episode = ql(
        hospital,
        featurisation = feature,
        gamma = 0.9,
        alpha = None,
        epsilon = 0.1,
        num_episodes = number_episodes,
        num_steps = number_steps)

# Run hospital with the policy learned above for number_steps steps
# Record allocations and plot bar plot
props, rewards, cured, time, cured_types = simulate(
        hospital,
        feature,
        Q_weights,
        steps = number_steps,
        plot = "hist")

print("\n\n\nFeature vector at the end of the simulation is: \n", feature(hospital))
print("\nQueues and doctors at the end of the simulation are:\n")
hospital.pretty_print()
print(props) 
print("\nThe average reward after the simulation with the fixed Q_weights is : {}\nThe whole list:\n{}"
      .format(np.mean(rewards), rewards))
print("\n{} patients were cured during the simulation of {} steps.\n".format(cured, number_steps))
print("Patients cured by types: \n{}\n".format(cured_types))
print("Total time waited by the cured patients: {}\n".format(time))
print("The final weight matrix is {}".format(Q_weights))

# Run hospital with the naive policy for number_steps steps
# Record allocations and plot heatmap
p_naive, r_naives = simulate_naive(hospital, steps = number_steps, plot = "map")
print("\nThe average reward after the simulation with naive policy is:",np.mean(r_naives))


