# import numpy as np
# import scipy.stats as sct
import random
from scipy.stats import binom, expon


class Hospital(object):
    """Defines a hospital with doctors of different types"""

    def __init__(self,
            occupancy,
            doctors,
            rate_arrivals):
        """
        Constructs a hospital

        parameters
        ----------
        occupancy     - Int (maximum capacity of hospital)
        doctors       - List of Doctors
        rate_arrivals - List of floats (how long it takes for patients of each type (0, 1, ...) to arrive, in minutes)
        """
        self.occupancy = occupancy

        self.state     = [self.newPatient] + self.doctors

        self.actions   = [None] + list(range(len(doctors)))

        # Make a dict out of it?
        self.rate_arrivals = rate_arrivals

        # How to define the transition function?
        # The state space is so huge
        #def t(self, s, a, s_):
        #    if self.state[0][0] == 0:
        #        return None
        #    else:
        #        need = self.state[0][1]
        #        
        ## Not sure if I'm doing the right thing here
        #def r(self, s, a, s_):
        #    if s[0][0] == 0:
        #        return 0

        #    # The hospital becomes overcrowded
        #    if sum(map(lambda x: len(x.queue), s)) == self.occupancy:
        #        return -1000
        #        # Can it be infinite?
        #        # return -np.inf

        #    need = s[0][1]
        #    result = 0

        #    # Try to allocate a patient to a doctor who cannot treat them
        #    # This could also be implemented directly, for simplicity
        #    if not s[a].can_treat(need):
        #        result += -1000
        #        # return -np.inf

        #    # Add up all the waiting times
        #    result += -sum(map(lambda x: sum(x.queue), s))
        #    
        #    return result

    # Not sure if the time period thing is a good idea
    def time_advance(self, policy, time_period):
        """
        Update the state (after a certain time period)

        Inputs:
        ----------
        policy - Function
        time_period - Float (how long to go forward)
        """
        s = self.state

        action = policy(s, self.actions)
        if action is not None:
            s[action+1].queue.append(s[0])

        # Look at which doctors finished during the time_period and update them
        for i, doctor in enumerate(s[1:]):
            doctor.update(time_period)

        # Look at if there is a new patient
        p = expon.cdf(time_period, scale = self.rate_arrivals)
        if binom.rvs(1, p):
            # Severity sampled from uniform
            s[0] = Patient(need = random.choice(self.actions[1:]))
        else:
            s[0] = None

        # Now do the rewards thing?

    def policy_random(self, state, actions):
        """The random policy"""
        if self.state[0]:
            # Return one of the possible actions with probability 1/(number of possible actions)
            return random.choice(actions[1:])
        return None

    def pretty_print(self):
        """Prints out the hospital state"""
        s = self.state

        # Make the state into a list of lists of waiting times
        new = [ list(map(lambda x: x.wait, d.queue)) for d in s[1:] ]
        lengths = list(map(len, new))

        longest_length = max(lengths)

        for i in range(len(lengths)):
            for _ in range(longest_length - lengths[i]):
                new[i].append(" ") 
        
        # Transpose, kind of
        new = list(zip(*new))

        format_row = "{:^10}" * len(lengths)
        print(format_row.format(*["Doctor "+str(i) for i in range(1,len(s)+1)]))
        # How to print a tuple?
        print(format_row.format(*[d.busy.need if d.busy else "None" for d in s[1:]]))
        print(("{:*^"+str(10*len(lengths))+"}").format(""))
        for row in new:
            print(format_row.format(*row))

        print("New patient:", s[0].need if s[0] else None,"\n")

class Doctor(object):
    """Defines a doctor"""

    def __init__(self,
            doctor_type,
            rate_doctor):
        """
        Constructs a doctor

        parameters
        ----------
        doctor_type - Int (doctor specialty)
        rate_doctor - Float (how long it takes on average to treat a patient)
        """
        self.type = doctor_type
        self.rate = rate_doctor
        self.queue = []
        # Patient currently being treated;
        # None if doctor is free
        self.busy = None
 

    def update(self, time_period):
        """Update the state of the doctor"""
        for p in self.queue: p.wait += 1
        # for p in self.queue: p.wait += time_period

        if self.busy:
            if binom.rvs(1, expon.cdf(time_period, scale = self.rate)): # If done
                self.busy = None
                # print("Done!")
            else:
                return # Don't change anything

        if self.queue: # If queue not empty
            self.busy = self.queue.pop(0)

    # Is it worth transforming that into a method that only works on Patient objects?
    def can_treat(self, severity):
        """Return whether the doctor can treat that type of ailment"""
        return severity <= self.type


class Patient(object):
    """Defines a patient"""

    def __init__(self,
        need):
        """
        Constructs a patient

        parameters
        ----------
        need - Int (severity of patient's ailment; higher is more urgent)
        """
        self.need = need
        self.wait = 0
