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
        

pi = [ 1/len(N) for i in range(len(N)) ]
# print(pi)

rng = np.random.default_rng()

# for k in range(int(hours*60/mins)):
#
#    # rng.choice(1, N(n), p=[ N[n]*L[n]*mins/60 for n in range(N(n)) ])
#
#    # arrival = np.random.bernoulli(l*mins/60)
#    # Which doctor type do they want?
#    doc = np.random.uniform(0,len(N))
#
#    for n in N:
#        for i in range(n):
#            # done = np.random.bernoulli(N(n)*L(n)*mins/60)
            

# [("GP", [15, 17]), ("nurse", [7, 8, 7]), ("eye doctor", [20]), ("physio", [30])]

# [(1, "any"), (5, [1, 1]), (0, [1, 1, 0]), (1, [1]), (2, [1])]



class Hospital(object):
    """Defines a hospital with doctors of different types"""

    def __init__(self,
            occupancy,
            doctor_types,
            num_doctors,
            rate_doctors,
            rate_arrivals):
        """
        Constructs a hospital

        parameters
        ----------
        occupancy     - Int (maximum capacity of hospital)
        doctor_types  - List of strings or ints (doctor specialties)
            # ints could represent the priority of each doctor
            # e.g. doctor type 0 can treat level 0 and below, type 1 can treat level 1 and below, etc...
        num_doctors   - List of ints (number of doctors of each type)
        rate_doctors  - List of floats (for each type, how long it takes on average to treat a patient)
            # Could make it more complicated by having a rate per doctor rather than per type
        rate_arrivals - List of floats (how many patients of each type (0, 1, ...) arrive per hour)
        """
        self.occupancy = occupancy

        self.types     = doctor_types
     
        # Arrival and then one empty queue per doctor
        self.state     = [(0, 0)] + [ [] for _ in range(sum(num_doctors)) ]

        self.action    = list(range(sum(num_doctors)))

        self.rates     = [rate_arrivals] + rate_doctors


    def time_advance(self, time_period):
        """
        Update the state (after a certain time period)

        Inputs:
        ----------
        """
        
def nest_increment(l):
    """ Add 1 to all elements in a nested list """
    # Do we need to make it usable with numpy arrays?
    if type(l) is not list:
        return l+1
    else:
        return list(map(nest_increment, l))
    

