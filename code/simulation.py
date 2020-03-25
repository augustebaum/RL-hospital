from hospital import *

# Two doctors
# One of level 0 that treats people in 3 min on average
# One of level 1 that treats people in 4 min on average
doctors = [Doctor(0, 3), Doctor(1, 4)]


# Hospital with occupancy of 20 people and one patient arriving every 1 min on average
hospital = Hospital(20, doctors, 1)

# Time-step
minutes = 1
limit = 8

#for i in range(8):
#    # Print state
#    print("After",i*minutes,"minutes:")
#    hospital.pretty_print()
#
#    # Simulate time-step of 1 minute
#    hospital.time_advance(hospital.policy_random, minutes)
#
#    print("\n")

for i in range(limit):
    # Simulate time-step of 1 minute
    hospital.time_advance(hospital.policy_random, minutes)

    hospital.pretty_print()

