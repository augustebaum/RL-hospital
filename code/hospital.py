import random
from scipy.stats import binom, expon
import numpy as np

class Hospital(object):
    """Defines a hospital with doctors of different types"""

    def __init__(self,
            occupancy,
            doctors,
            needs):
        """
        Constructs a hospital

        parameters
        ----------
        occupancy     - Int (maximum capacity of hospital)
        doctors       - List of Doctors
        needs         - List of floats (probabilities of patients of each type (0, 1, ...) to arrive -- will be normalised)
        """
        self.occupancy = occupancy

        self.actions   = sorted([*{*[d.type for d in doctors]}])

        # Normalised needs; doctor types start at 0
        if len(needs) != len(self.actions):
            raise ValueError("There aren't the same number of doctor types and patient needs!\n"+"Number of types: "+str(len(self.actions))+"\n"+"Number of needs: "+str(len(needs)))

        self.needs     = needs
        
        # Each list corresponds to a type of doctor
        # Will be assuming that doctor types go 0, 1, 2...
        self.queues = [[] for _ in range(len(self.actions)) ]
        # [[]]*len(self.actions) produces unexpected behaviour, try to append something to one sublist!

        self.doctors = doctors

        # Initialise patient with a random need
        self.newPatient = Patient(need = random.choices(self.actions, weights = self.needs)[0])

    def update(self, policy):
        """
        Update the state

        Inputs:
        ----------
        policy - Function that dispatches a patient based on the state
        """
        s = self
        # terminate = False
        reward = 0

        for queue in s.queues:
            for patient in queue: patient.update()

        action = policy()
        s.queues[action].append(s.newPatient)

        for d in s.doctors:
            d.update()
            if not(d.busy):
                queue = self.queues[d.type]
                # if queue is not empty
                if queue:
                # If free and queue not empty
                    d.busy = queue.pop(0)
                    if not(d.can_treat(d.busy.need)):
                        reward -= 1000
                        # What do you do with the failed patient? Right now they're just chucked away
                        d.busy = None

        # Now do the rewards thing
        if sum(map(len, self.queues)) >= self.occupancy:
            reward -= 100
            # terminate = True

        s.newPatient = Patient(need = random.choices(self.actions, weights = self.needs)[0])

        # return terminal, reward
        
    ##############################################
    #  just added this function which is similar #
    #  to the previous one                       #
    ##############################################
    def next_step(self, action):
        """
        Update the internal state for sarsa and get the reward

        Inputs:
        ----------
        action - the current action to be taken
        """
        reward = 0

        for queue in self.queues:
            for patient in queue: 
                patient.update()

        self.queues[action].append(self.newPatient)

        for d in self.doctors:
            d.update()
            # If you get a patient you can't treat, send them away and become free
            #if not(d.busy):
            #    queue = self.queues[d.type]
            #    # if queue is not empty
            #    if queue:
            #    # If free and queue not empty
            #        d.busy = queue.pop(0)
            #        if not(d.can_treat(d.busy.need)):
            #            reward -= d.busy.need
            #            # What do you do with the failed patient? Right now they're just chucked away
            #            d.busy = None
            #        else:
            #            reward += 1
            # If you get a patient you can't treat, send them away and try again immediately, as many times as possible
            while not(d.busy) and self.queues[d.type]:
                queue = self.queues[d.type]
                d.busy = queue.pop(0)
                if not(d.can_treat(d.busy.need)):
                # if patient can't be treated
                    reward -= 10*(d.busy.wait + 1)*d.busy.need
                    d.busy = None
                else:
                    reward -= (d.busy.wait + 1)*d.busy.need

        # More people is bad
        reward -= sum(map(len, self.queues))

        if sum(map(len, self.queues)) >= self.occupancy:
        # if hospital is full
            # reward -= (sum(map(len, self.queues)) - self.occupancy)
            reward -= 1000
            self.isTerminal = True

        self.newPatient = Patient(need = random.choices(self.actions, weights = self.needs)[0])

        return reward

    def simulate(self, policy, limit = 30):
        s = self
        for _ in range(limit):
            # s.pretty_print()
            print(s.feature())
            s.update(policy)
            # terminal, reward = s.update(s.policy)
            # if terminal: break
        

    def reset(self):
        self.queues = [[] for _ in range(len(self.actions)) ]
        self.newPatient = Patient(need = random.choices(self.actions, weights = self.needs)[0])
        for d in self.doctors:
            d.busy = None
        self.isTerminal = False


##### Make it nice ############################
    def pretty_print(self):
        """Prints out the hospital state"""
        s = self

        for i, q in enumerate(s.queues):
            print("Queue of type", i,":\t", [p.wait for p in q])

        for d in s.doctors:
            print("Doctor of type", d.type,":\t", d.busy.need if d.busy else None)

        print("New patient with need:", s.newPatient.need, "\n")


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
        rate_doctor - Float (probability of being done with a patient at each timestep)
        """
        self.type = doctor_type
        self.rate = rate_doctor
        # Patient currently being treated;
        # None if doctor is free
        self.busy = None

    def update(self):
        """Update the state of the doctor"""
        if self.busy and binom.rvs(1, self.rate): # If done
            self.busy = None
<<<<<<< HEAD
=======
            #print("done")
>>>>>>> 4df58fd25fcffbc5d8d8fb19fb946c0f683f74bd

    def can_treat(self, severity):
        """Return whether the doctor can treat that type of ailment"""
        return severity <= self.type


class Patient(object):
    """Defines a patient"""

    def __init__(self,
        need, wait = 0):
        """
        Constructs a patient

        parameters
        ----------
        need - Int (severity of patient's ailment; higher is more urgent)
        """
        self.need = need
        self.wait = wait

    def update(self):
        self.wait += 1
