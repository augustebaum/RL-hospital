import random
from scipy.stats import binom, expon

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
            if not(d.busy) and (queue := self.queues[d.type]):
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

    def simulate(self, policy, limit=30):
        s = self
        for _ in range(limit):
            # s.pretty_print()
            print(s.feature())
            s.update(policy)
            # terminal, reward = s.update(s.policy)
            # if terminal: break

##### FEATURISATIONS ##########################
    def feature(self):
        """A representation of the hospital"""
        res = []
        # Number of patients waiting in hospital
        res.append(sum(map(len, self.queues)))
        # Whether or not a given queue has more or fewer patients with a certain need than a threshold
        for q in self.queues:
            q = [ p.need for p in q ]             
            for i in self.actions:
                res.append(int(q.count(i) >= 3))
        return res

##### POLICIES ################################  
    def policy_random(self):
        """The random policy"""
        # Return one of the possible actions with probability 1/(number of possible actions)
        return random.choice(self.actions)

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

    def update(self):
        self.wait += 1
