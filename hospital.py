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
        needs         - List of floats (probabilities of patients of each type (0, 1, ...) to arrive -- while be normalised)
        """
        self.occupancy = occupancy

        self.actions   = sorted([*{*[d.type for d in doctors]}])

        # Normalised needs; doctor types start at 0
        if len(needs) != len(self.actions):
            raise ValueError("There aren't the same number of doctor types and patient needs!\n"+"Number of types: "+str(len(self.actions))+"\n"+"Number of needs: "+str(len(needs)))

        self.needs     = needs
        
        # Each list corresponds to a type of doctor
        # Will be assuming that doctor types go 0, 1, 2...
        self.state     = [[]]*len(self.actions) + doctors + [Patient(need = random.choices(self.actions, weights = self.needs)[0])] 

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

    def time_advance(self, policy):
        """
        Update the state (after a certain time period)

        Inputs:
        ----------
        policy - Function that dispatches a patient based on the state
        """
        s = self.state

        # Increase waiting time by 1
        for queue in s[:len(self.actions)]:
            queue = [ p.update() for p in queue ]
            print("Queue updated")

        # Make decision
        action = policy()
        s[action].append(s[-1])
        print("Queue",action,"now looks like", [p.wait for p in s[action]])

        # Update doctors
        for i, doctor in enumerate(s[len(self.actions):-1]):
            doctor.update()
            print("Doctor",i)
            if (doctor.busy is None) and (queue := self.state[doctor.type]):
            # BUG: doctor of 0 takes patients from queue 1!!
            # If free and queue not empty
                print("Will take from queue", [ p.wait for p in queue ])
                doctor.busy = queue.pop(0)
                if not(doctor.can_treat(doctor.busy.need)):
                    print("Doctor: 'Can't treat them!'")
                    # reward = -1000
                    doctor.busy = None
                print("Done, taking someone else")

        # Now do the rewards thing

        # New patient
        s[-1] = Patient(need = random.choices(self.actions, weights = self.needs)[0])


    def policy_random(self):
        """The random policy"""
        # Return one of the possible actions with probability 1/(number of possible actions)
        return random.choice(self.actions)

    def pretty_print(self):
        """Prints out the hospital state"""
        s = self.state

        for i, l in enumerate(s[:len(self.actions)]):
            print("Queue of type", i,":\t", [p.wait for p in l])

        for d in s[len(self.actions):-1]:
            print("Doctor of type", d.type,":\t", d.busy.need if d.busy else None)
        # Make the state into a list of lists of waiting times
        # new = [ list(map(lambda x: x.wait, d.queue)) for d in s[1:] ]
        # lengths = list(map(len, new))

        # longest_length = max(lengths)

        # for i in range(len(lengths)):
        #     for _ in range(longest_length - lengths[i]):
        #         new[i].append(" ") 
        # 
        # # Transpose, kind of
        # new = list(zip(*new))

        # format_row = "{:^10}" * len(lengths)
        # print(format_row.format(*["Doctor "+str(i) for i in range(1,len(s)+1)]))
        # # How to print a tuple?
        # print(format_row.format(*[d.busy.need if d.busy else "None" for d in s[1:]]))
        # print(("{:*^"+str(10*len(lengths))+"}").format(""))
        # for row in new:
        #     print(format_row.format(*row))

        print("New patient with need:", s[-1].need, "\n")

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
        self.queue = []
        # Patient currently being treated;
        # None if doctor is free
        self.busy = None
 

    def update(self):
        """Update the state of the doctor"""
        # for p in self.queue: p.wait += 1
        # for p in self.queue: p.wait += time_period

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
