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
number_steps = 1000
number_episodes = 100

t_list, Q_weights, total_reward_per_episode = ql(hospital, featurisation = feature, gamma = 0.9, alpha = None, epsilon = 0.1, num_episodes = number_episodes, num_steps = number_steps)

props, rewards = simulate(hospital, feature, Q_weights, steps = 20, epsilon = 0.1, plot = "hist", printSteps = 10)
print(feature(hospital))
hospital.pretty_print()
print(Q_weights)
print(np.mean(rewards))

#### will print the simulation rewards against random rewards
p_naive, r_naives = simulate_naive(hospital, steps = 100, plot = "map")
print(np.mean(r_naives))


