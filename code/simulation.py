from hospital import *

# Doctors
# One of level 0 that has probability of being done of 0.2
# One of level 0 that has probability of being done of 0.1
# One of level 1 that has probability of being done of 0.1
# One of level 2 that has probability of being done of 0.05
doctors = [Doctor(0, 0.1),
           Doctor(0, 0.1),
           Doctor(1, 0.1),
           Doctor(2, 0.05),
           Doctor(2, 0.1)]

types = [2, 1, 1]
probs = [0.2, 0.1, 0.1, 0.05]

# Hospital with occupancy of 20 people 
# Patient of type 0 five times as likely than of type 2
# Patient of type 1 twice as likely than of type 2
hospital = Hospital(20, doctors, [5, 2, 1])

hospital.simulate(sample_from_epsilon_greedy, episodes = 300,limit = 40)
hospital.simulate(hospital.policy_random, episodes = 270,limit = 40)
