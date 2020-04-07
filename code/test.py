from hospital import *
from simulation import *
from learning import *
import numpy as np
import matplotlib.pyplot as plt

doctors = [Doctor(0, 0.1), Doctor(1, 0.1), Doctor(2, 0.9)]
hospital = Hospital(20, doctors, [1, 1, 1])

# alpha was previously 1/50, now is calculated as 1/num_steps
t_list, Q_weights, total_reward_per_episode = sarsa(hospital, featurisation = feature_1, gamma = 0.9, alpha = None, epsilon = 0.1, num_episodes = 50, num_steps = 50)

props = simulate(hospital, feature_1, Q_weights, steps = 10000, plot = True)
print(props)
