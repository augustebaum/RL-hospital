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
           Doctor(6, 0.1),
           Doctor(7, 0.1),
           Doctor(8, 0.9)]
hospital = Hospital(2000, doctors, [1, 1, 1, 1, 1, 1, 1, 1, 1])

# doctors = [Doctor(0, 0.5),
           # Doctor(1, 0.5)]
# hospital = Hospital(20, doctors, [1, 1])

feature = feature_5

t_list, Q_weights, total_reward_per_episode = ql(hospital, featurisation = feature, alpha = None, gamma = 0.85, epsilon = 0.9, num_episodes = 25, num_steps = 500)

props = simulate(hospital, feature, Q_weights, steps = 1000, epsilon = 0.9, plot = "hist")
print(feature(hospital))
hospital.pretty_print()
print(props)

simulate_naive(hospital, plot = 'hist')
hospital.pretty_print()
