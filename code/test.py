from hospital import *
from simulation import *
from learning import *
import numpy as np
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
hospital = Hospital(20, doctors, [1, 1, 1, 1, 1, 1, 1, 1, 1])

feature = feature_1

t_list, Q_weights, total_reward_per_episode, timeline_episodes = sarsa(hospital, featurisation = feature, gamma = 0.85, alpha = 1/100, epsilon = 0.3, num_episodes = 50, num_steps = 100)

props = simulate(hospital, feature, Q_weights, steps = 10000, plot = True)
print(feature_1(hospital))
hospital.pretty_print()
print(props)
