from hospital import *

# Two doctors
# One of level 0 that treats people in 3 min on average
# One of level 1 that treats people in 4 min on average
doctors = [Doctor(0, 0.2), Doctor(1, 0.1)]

# Hospital with occupancy of 20 people and patient of type 0 twice as likely than of type 1
hospital = Hospital(20, doctors, [2, 1])

limit = 4
for i in range(limit):
    hospital.pretty_print()
    hospital.time_advance(hospital.policy_random)

