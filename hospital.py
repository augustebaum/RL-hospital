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
        # self.state     = [(0, 0)] + doctors
        self.state     = doctors

        # self.actions   = list(range(len(doctors)))
        # self.actions   = [None] + list(range(1, len(doctors)))
        self.actions   = [None] + list(range(len(doctors)))

        self.rate_arrivals = rate_arrivals
        # self.rates     = [rate_arrivals] + rate_doctors

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
    def time_advance(self, policy):
        """
        Update the state (after a certain time period)

        Inputs:
        ----------
        policy - Function
        """
        ### NEED TO FIGURE OUT THE ORDER OF THE ACTIONS
        # Look at which doctors were done
        for doctor in self.state:
            doctor.update()

        # Look at if there is a new patient
        if new_patient := np.random.binomial(1, rate_arrivals):
            # Check the rate stuff
            self.state[0] = Patient(np.random.uniform(0, 3))
            print("there is a new patient with priority", patient_need)
        else:
            self.state[0] = None

        action = policy(state, self.actions)
        print("Take action:", action)

    # Should this be a class method?
    def policy_random(state, actions):
        """
        The random policy
        """
        if state[0] is None:
            return None
        else:
            pass
            # Return one of the possible actions with probability 1/number of possible actions


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

        # Patient currently being treated, and None if the doctor is free
        self.busy = None


    def update(self):
        """Update the state of the doctor"""
        # Could probably be shortened
        for i in range(len(self.queue)): self.queue[i].wait += 1

        if self.busy:
            # Will have to translate rate into probability

            if is_done := np.random.binomial(1, self.rate):
                if self.queue:
                    self.busy = self.queue.pop(0)
                else:
                    self.busy = None

    # Is it worth transforming that into a method that only works on Patient objects?
    def can_treat(self, severity):
        """ Return whether the doctor can treat that type of patient """
        return severity <= self.type


class Patient(object):
    """Defines a patient"""

    def __init__(self,
        need):
        """
        Constructs a patient

        parameters
        ----------
        need - Int (how severe this patient's ailment is; higher is more urgent)
        """
        self.need = need

        self.wait = 0
