import random
from scipy.stats import binom, expon
import numpy as np
from math import *


class Hospital(object):
    """Defines a hospital with doctors of different types"""

    def __init__(self, occupancy, doctors, needs):
        """
        Constructs a hospital

        parameters
        ----------
        occupancy     - Int (maximum capacity of hospital)
        doctors       - List of Doctors
        needs         - List of floats (probabilities of patients of each type (0, 1, ...) to arrive -- will be normalised)
        """
        self.occupancy = occupancy

        self.actions = sorted([*{*[d.type for d in doctors]}])

        # Normalised needs; doctor types start at 0
        if len(needs) != len(self.actions):
            raise ValueError(
                "There aren't the same number of doctor types and patient needs!\n"
                + "Number of types: "
                + str(len(self.actions))
                + "\n"
                + "Number of needs: "
                + str(len(needs))
            )

        self.needs = needs

        # Each list corresponds to a type of doctor
        # Will be assuming that doctor types go 0, 1, 2...
        self.queues = [[] for _ in range(len(self.actions))]

        self.doctors = doctors

        # Initialise patient with a random need
        self.newPatient = Patient(
            need=random.choices(self.actions, weights=self.needs)[0]
        )

        # a list that will keep the cured patients
        self.cured = []

    def next_step(self, action, checkBefore=True, cap_penalty=False):
        """
        Update the internal state according to an action and get the reward

        Inputs:
        ----------
        action - the current action to be taken
        checkBefore - Whether misallocation is penalized at the beginning (default) or end of the queue
        cap_penalty - Whether agent is penalized for reaching the capacity of the hospital
        """
        reward = 0
        s = self

        # Increment waits for each patient
        for queue in s.queues:
            for patient in queue:
                patient.update()

        # assign the next patient to one of the queues given the current action
        s.queues[action].append(s.newPatient)

        # If the patient was misallocated and this behaviour is selected, immediately issue penalty
        if checkBefore and s.newPatient.need > action:
            reward -= 50 * s.newPatient.need

        for d in s.doctors:
            d.update()
            # If you get a patient you can't treat, send them away and become free
            # d.busy = None evaluates to False
            if not (d.busy):
                queue = s.queues[d.type]
                # if queue is not empty take the first person and let them visit the doctor
                if queue:
                    d.busy = queue.pop(0)
                    # Penalty proportional to time waited
                    reward -= d.busy.need * d.busy.wait
                    if not (d.can_treat(d.busy.need)):
                        if not (checkBefore):
                            reward -= d.busy.need * 50
                        d.busy = None
                    else:
                        # Reward for curing patients
                        reward += 100
                        # Alternatively:
                        # reward += (d.busy.need + 1) * 30
                        # add the current patient to the `cured` list
                        self.cured.append(d.busy)
        # Below we could add penalties for having many people in the queues
        # these rewards will realise on each step so we must not make them too great
        ######
        # More people is bad, so is waiting a long time
        # Option 1:
        # reward -= sum(map(len, s.queues))
        # Option 2:
        # for q in s.queues:
        #     reward -= sum([ (p.wait + 1)*p.need for p in q])

        # below we define the penalty for reaching the capacity of the hospital
        # and we also terminate the current episode
        if sum(map(len, s.queues)) >= s.occupancy:
            # defines a penalty for reaching the total capacity of the hospital
            if cap_penalty:
                reward -= 10_000

            s.isTerminal = True

        # set the next new patient
        s.newPatient = Patient(need=random.choices(s.actions, weights=s.needs)[0])

        return reward

    # a function to approximate the maximum possible reward in the simulation
    # Careful, this is model dependent!
    def max_average_reward(self):
        need_list = np.array(self.actions) + 1
        arrival_probs = np.array(self.needs) / sum(self.needs)
        return np.dot(need_list, arrival_probs)

    def reset(self):
        self.queues = [[] for _ in range(len(self.actions))]
        self.cured = []
        self.newPatient = Patient(
            need=random.choices(self.actions, weights=self.needs)[0]
        )
        for d in self.doctors:
            d.busy = None
        self.isTerminal = False

    ##### Make it nice ############################
    def pretty_print(self):
        """Prints out the hospital state"""
        s = self

        for i, q in enumerate(s.queues):
            print("Queue of type", i, ":\t", [p.wait for p in q])

        for d in s.doctors:
            print("Doctor of type", d.type, ":\t", d.busy.need if d.busy else None)

        print("New patient with need:", s.newPatient.need, "\n")


class Doctor(object):
    """Defines a doctor"""

    def __init__(self, doctor_type, rate_doctor):
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
        # None means doctor is free
        self.busy = None

    def update(self):
        """Update the state of the doctor"""
        if self.busy and binom.rvs(1, self.rate):  # If busy, flip a coin
            self.busy = None

    def can_treat(self, severity):
        """Return whether the doctor can treat that type of ailment"""
        return severity <= self.type


class Patient(object):
    """Defines a patient"""

    def __init__(self, need, wait=0):
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
