from hospital import *
from simulation import *
from learning import *
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

doctors = [Doctor(0, 0.1),
           Doctor(1, 0.1),
           Doctor(2, 0.1),
           Doctor(3, 0.1),
           Doctor(4, 0.1),
           Doctor(5, 0.1),
           Doctor(6, 1)]
# watchout the capacity below might be changed
hospital = Hospital(20, doctors, [10000, 1, 1, 1, 1, 1, 1])

feature = feature_5
number_steps = 100
number_episodes = 300

t_list, Q_weights, total_reward_per_episode = sarsa(hospital, featurisation = feature, gamma = 0.9, alpha = None, epsilon = 0.1, num_episodes = number_episodes, num_steps = number_steps)

props = simulate(hospital, feature, Q_weights, steps = number_steps, plot = "hist")
print(feature_1(hospital))
hospital.pretty_print()
print(props)

#### will print the simulation rewards against random rewards
