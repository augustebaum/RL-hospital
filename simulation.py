from hospital import *
# np.random.seed(1234)

# Two doctors
# One of level 0 that treats people in 10 min on average
d1=Doctor(0, 10)
# One of level 1 that treats people in 15 min on average
d2=Doctor(1, 15)
# doctors = [Doctor(0, 10), Doctor(1, 15)]
doctors = [d1, d2]

# Hospital with occupancy of 20 people and one patient arriving every 3 min on average
hospital = Hospital(20, doctors, 3)

for i in range(5):
    print("Step",i)
    hospital.pretty_print()
    # Simulate time-step of 1 minute
    hospital.time_advance(hospital.policy_random, 1)
    print("\n")
