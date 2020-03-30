import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, expon
from learning    import *

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
        
        
        ##########################################################
        self.reward = 0  ### intended to be the reward per episode
        ##########################################################


        self.actions   = sorted([*{*[d.type for d in doctors]}])
        
        #########################################################
        #           adding the weight matrix here               #
        #   one weight vector for each action with the dimension
        #   of the feaature vector
        
        num_unique_doctors = len(self.actions)
        self.q_weights = np.ones((num_unique_doctors, num_unique_doctors**2 + 1))
        #self.q_weights = [[1 for _ in range(1 + num_unique_doctors**2)] for _ in range(num_unique_doctors)]
        ########################################################
        self.is_terminal = False
        
        # Doctor types start at 0
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
    
        reward = 0

        for queue in s.queues:
            for patient in queue: 
                patient.update()
        
        action = policy(s.feature(), s.q_weights)
        #print(action)
        s.queues[action].append(s.newPatient)

        for d in s.doctors:
            d.update()
            if not(d.busy) and s.queues[d.type]:
            # If free and queue not empty
                d.busy = s.queues[d.type].pop(0)
                patient = d.busy
                # Take next in line
                if not(d.can_treat(patient)):
                    # If d can't treat that patient
                    reward -= patient.need
                    # What do you do with the failed patient? Right now they're just chucked away
                    d.busy = None
                else:
                    reward -= (d.type - patient.need)

        # Now do the rewards thing
        if sum(map(len, s.queues)) >= s.occupancy:
            reward -= 1000
            s.is_terminal = True

        s.newPatient = Patient(need = random.choices(s.actions, weights = s.needs)[0])
        s.reward += reward

        return reward
        

    def simulate(self, policy, episodes = 1, limit=20):
        s = self
        
        #############################
        total_reward_per_step    = np.zeros(limit)
        total_reward_per_episode = np.zeros(episodes)
        timeline                 = [i for i in range(limit)]
        timeline_episodes        = [i for i in range(episodes)]
        
        #############################################
        ######## trying to iplement SARSA below  ####
        
        for i in range(episodes):
            current_state = self.feature()
            current_action = policy(current_state, self.q_weights)
            
            for step in range(limit):
                
                #print("At time {}".format(step))
                #print("\tThe feature vector is: {}".format(s.feature()))
                #s.pretty_print()
                step_reward = s.update(policy)
                next_state = self.feature()
                next_action = policy(next_state, self.q_weights)
                self.q_weights = sarsa_update(
                        self.q_weights,
                        current_state,
                        current_action,
                        step_reward,
                        next_state,
                        next_action,
                        gamma = 0.8,
                        alpha = 1/limit
                        )
                current_state = next_state
                current_action = next_action
                
                ### just follows the reward update at each step - will get rid of it
                if step == 0:
                    total_reward_per_step[step] = step_reward
                else:
                    total_reward_per_step[step] = total_reward_per_step[step - 1] + step_reward
               
                # terminal, reward = s.update(s.policy)
            total_reward_per_episode[i] = total_reward_per_step[-1] 
            self.reset() #### reset the data before the next episode
            
        print("\nq_weights:\n", self.q_weights)
        #plt.figure(1)
        #plt.plot(timeline, total_reward_per_step)
        plt.figure(2)
        plt.plot(range(episodes), total_reward_per_episode)
    

    def reset(self):
        self.queues = [[] for _ in range(len(self.actions)) ]
        self.newPatient = Patient(need = random.choices(self.actions, weights = self.needs)[0])
        for d in self.doctors:
            d.busy = None

        # reset the reward for the episode to 0 
        self.reward = 0 

##### FEATURISATIONS ##########################
    def feature(self):
        """A representation of the hospital"""
        res = []
        # Number of patients waiting in hospital
        res.append(sum(map(len, self.queues))/len(self.queues))
        # Whether or not a given queue has more or fewer patients with a certain need than a threshold
        for q in self.queues:
            q = [ p.need for p in q ]             
            for i in self.actions:
                res.append(int(q.count(i) >= 2))
        return np.array(res)

##### Make it nice ############################
    def pretty_print(self):
        """Prints out the hospital state"""
        s = self

        for i, q in enumerate(s.queues):
            print("\tQueue of type", i,":\t", [(p.need, p.wait) for p in q])
            # printing tuples -> 1st patients's need' 2nd wait time

        for d in s.doctors:
            print("\tDoctor of type", d.type,":\t", d.busy.need if d.busy else None)

        print("\tNew patient with need:", s.newPatient.need, "\n")


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
