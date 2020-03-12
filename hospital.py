import numpy as np

class Hospital(object):
    """Defines a hospital with doctors of different types"""

    def __init__(self,
            occupancy,
            # doctor_types,
            # num_doctors,
            # rate_doctors,
            doctors,
            rate_arrivals):
        """
        Constructs a hospital

        parameters
        ----------
        occupancy     - Int (maximum capacity of hospital)
        doctors       - List of Doctors
        doctor_types  - List of strings or ints (doctor specialties)
            # ints could represent the priority of each doctor
            # e.g. doctor type 0 can treat level 0 and below, type 1 can treat level 1 and below, etc...
        num_doctors   - List of ints (number of doctors of each type)
        rate_doctors  - List of floats (for each type, how long it takes on average to treat a patient)
            # Could make it more complicated by having a rate per doctor rather than per type
        rate_arrivals - List of floats (how many patients of each type (0, 1, ...) arrive per hour)
        """
        self.occupancy = occupancy

        # self.types     = doctor_types
     
        # Redefine in terms of Doctor instances
        # Arrival and then one empty queue per doctor
        # self.state     = [(0, 0)] + [ [] for _ in range(sum(num_doctors)) ]
        self.state     = [(0, 0)] + doctors

        # self.actions   = list(range(len(doctors)))
        self.actions   = [None] + list(range(1, len(doctors)))

        # self.rates     = [rate_arrivals] + rate_doctors
        # How to define the transition function?
        # The state space is so huge
        def t(self, s, a, s_):
            if self.state[0][0] == 0:
                return None
            else:
                need = self.state[0][1]
                
        # Not sure if I'm doing the right thing here
        def r(self, s, a, s_):
            if s[0][0] == 0:
                return 0

            # The hospital becomes overcrowded
            if sum(map(lambda x: len(x.queue), s)) == self.occupancy:
                # return -1000
                # Can it be infinite?
                return -np.inf

            need = s[0][1]
            result = 0

            # Try to allocate a patient to a doctor who cannot treat them
            if not s[a].can_treat(need):
                result += -1000
                # return -np.inf

            # Add up all the waiting times
            result += -sum(map(lambda x: sum(x.queue), s))
            
            return result

    # Not sure if the time period thing is a good idea
    def time_advance(self, state, time_period):
        """
        Update the state (after a certain time period)

        Inputs:
        ----------
        """
        pass

        
#def nest_increment(l):
#    """
#    Add 1 to all elements in a nested list
#        
#    Used to increases the waiting time of each patient in the hospital by 1
#    """
#    # Do we need to make it usable with numpy arrays?
#    if type(l) is not list:
#        return l+1
#    else:
#        return list(map(nest_increment, l))
    

class Doctor(object):
    """Defines a doctor"""

    def __init__(self,
            doctor_type,
            rate_doctor):
        """
        Constructs a doctor

        parameters
        ----------
        doctor_type - String or int (doctor specialty)
            # ints could represent the priority of doctor
            # e.g. doctor type 0 can treat level 0 and below, type 1 can treat level 1 and below, etc...
            # That idea is growing on me
        rate_doctor - Float (how long it takes on average to treat a patient)
        """
        self.type = doctor_type

        self.rate = rate_doctor

        self.queue = []

        self.occupied = False


    def done(self):
        """ Update the state when the doctor is done """
        # if the queue is not empty
        if self.queue:
            return self.queue.pop(0)

    def can_treat(self, severity):
        """ Return whether the doctor can treat that type of patient """
        return severity <= self.type
