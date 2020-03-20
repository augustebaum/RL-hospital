from hospital import *

# Two doctors
# One of level 0 that has probability of being done of 0.2
# One of level 1 that has probability of being done of 0.1
doctors = [Doctor(0, 0.2), Doctor(1, 0.1)]

# Hospital with occupancy of 20 people and patient of type 0 twice as likely than of type 1
hospital = Hospital(20, doctors, [2, 1])

limit = 30
for i in range(limit):
    hospital.pretty_print()
    hospital.update(hospital.policy_random)

