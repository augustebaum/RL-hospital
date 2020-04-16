from hospital import *
from simulation import *
from learning import *
import numpy as np
from matplotlib import rc
# rc('text', usetex=True)
import matplotlib.pyplot as plt

doctors = [Doctor(0, 0.1),
           Doctor(1, 0.1)]
# watchout the capacity below might be changed
hospital = Hospital(20000, doctors, [1, 1])

feature = feature_7
number_steps = 100
number_episodes = 100

t_list, Q_weights, total_reward_per_episode = ql(hospital, featurisation = feature, gamma = 0.9, alpha = None, epsilon = 0.1, num_episodes = number_episodes, num_steps = number_steps)

props, rewards, cured, time, cured_types = simulate(hospital, feature, Q_weights, steps = number_steps, plot = "hist")

print("\n\n\nFeature vector at the end of the simulation is: \n", feature_1(hospital))
print("\nQueues and doctors at the end of the simulation are:\n")
hospital.pretty_print()
print(props) 
print("\nThe final reward after the simulation with the fixed Q_weights is : {}\nthe whole list:\n{}"
      .format(sum(rewards), rewards))
print("\n{} patients are cured during the simulation of {} steps.\n".format(cured, number_steps))
print("Patients cured by types: \n{}\n".format(cured_types))
print("Total time waited by the cured patients: {}\n".format(time))
print("The final weight matrix is {}".format(Q_weights))

#### will print the simulation rewards against random rewards
p_naive, r_naives = simulate_naive(hospital, steps = 100, plot = "map")
print(np.mean(r_naives))


