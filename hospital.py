import numpy as np

# Mean arrivals per hour
l = 10

# Mean service times per hour
# for doctors of type 0, 1, 2, ...
L = [8, 5]

# Number of doctors of type 0, 1, 2, ...
N = [3, 1]

# Discretisation (in minutes)
mins = 5

hours = 0.5

# Poisson process lasting for *hours* hours (count every *mins* minutes)
arrivals = np.random.poisson(l*mins/60, size = int(hours*60/mins))

queue = np.zeros(len(N))
state = np.array([np.zeros(N[n]) for n in range(len(N))])

# Initialize with unbiased policy
def pi(s, a):
    # If no arrival then do nothing
    if s[0] == 0: return None
    else:
        

pi = [ 1/len(N) for i in range(len(N)) ]
print(pi)

rng = np.random.default_rng()

for k in range(int(hours*60/mins)):

    # rng.choice(1, N(n), p=[ N[n]*L[n]*mins/60 for n in range(N(n)) ])

    arrival = np.random.bernoulli(l*mins/60)
    # Which doctor type do they want?
    doc = np.random.uniform(0,len(N))

    for n in N:
        for i in range(n):
            done = np.random.bernoulli(N(n)*L(n)*mins/60)

    decision = np.

# [("GP", [15, 17]), ("nurse", [7, 8, 7]), ("eye doctor", [20]), ("physio", [30])]

# [(1, "any"), (5, [1, 1]), (0, [1, 1, 0]), (1, [1]), (2, [1])]



class Hospital(object):
    # Defines a hospital with doctors of different types

    def __init__(self,
            doctor_types,
            num_doctor,
            rate_doctor,
            rate_arrivals):
    """
    Constructs a hospital

    parameters
    ----------
    doctor_types  - List of strings or ints (doctor specialties)
    num_doctor    - List of ints (number of doctors of each type)
    rate_doctor   - List of floats (for each type, how long it takes on average to treat a patient)
    rate_arrivals - Float (how many patients arrive per hour)
    """
    self.types = doctor_types
 
    self.state = [(0, 0)] + [

    self.rates = [rate_arrivals] + rate_doctor

    # Who an incoming patient wants to see
    self.need  = ["any"] + doctor_types
    # self.policy = 
    

