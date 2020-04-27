"""
Part of this code has been adapted from code provided by Luke Dickens on the UCL module INST0060: Foundations of Machine Learning and Data Science
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from math import *

##### FEATURISATIONS ##########################
# Need to find featurisations with more elements -- or use basis functions
def feature_1(env):
    """A representation of the hospital"""
    res = []
    res.append(env.newPatient.need)   
    for q in env.queues:
        q = [ p.need for p in q ]             
        for i in env.actions:
            num_patients = q.count(i)
            if num_patients == 0: 
                res.append(0)
            elif num_patients < 3: 
                res.append(1)
            else:
                res.append(2)
  
    return np.array(res)

def feature_2(env):
    """A representation of the hospital"""
    res = []
    res.append(sum(map(len, env.queues))/len(env.queues))    
    for q in env.queues:
        q = [ p.need for p in q ]             
        for i in env.actions:
            res.append(int(q.count(i) >= 2))
    return np.array(res)

def feature_3(env):
    """A representation of the hospital
       The first element is just the total number of patients in all the queues.
       The following elements just represent the number of different patients(needs)
       in each queue.
    """
    res = []
    # Average number of patients waiting in the different queues
    res.append(sum(map(len, env.queues)) / len(env.queues))    
    # adds total number of different patients in separate queues
    for q in env.queues:
        q = [ p.need for p in q ]             
        for i in env.actions:
            res.append(q.count(i))
    return np.array(res)

def feature_4(env):
    """A representation of the hospital
       The first element is just the total number of patients in all the queues.
       The following elements include waiting time and number of different patient in 
       each queue.
    """
    res = []
    # Average number of patients waiting in the different queues
    res.append(sum(map(len, env.queues)) / len(env.queues))    
    # adds waiting time in the queue in the feature along with the number of patients
    # wait time added should be adjusted when changing the number of steps in the model
    for q in env.queues:
        q_ = [ p.need for p in q ]
        wait_time = sum([p.wait for p in q])/100 # log(sum([p.wait for p in q]) + e)
        res.append(wait_time)
        for i in env.actions:
            res.append(q_.count(i))
    return np.array(res)

def feature_5(env):
    """A representation of the hospital

       The first element is the need of the newly arrived patient.
       The second element is the total number of patients in all the queues.
       The following elements include average need and waiting time in 
       each queue.
    """
    res = []
    # Average number of patients waiting in the different queues
    res.append(sum(map(len, env.queues)) / len(env.queues))    
    # adds waiting time in the queue in the feature along with the number of patients
    # wait time added should be adjusted when changing the number of steps in the model
    for q in env.queues:
        # Something about the length of the queue perhaps?
        res.append(np.mean([ p.need for p in q ]) if q else 0)
        res.append(np.mean([ p.wait for p in q ]) if q else 0)
    # Prepend one-hot vector with new patient's need
    return np.concatenate((np.array(np.arange(len(env.actions)) == env.newPatient.need, dtype=int), np.array(res)))

def feature_6(env):
    """A representation of the hospital
       The first element is the need of the newly arrived patient.
       The second element is the total number of patients in all the queues.
       The following elements include average need and waiting time in 
       each queue.
    """
    res = []
    # Need of new patient
    # res.append(env.newPatient.need)
    # Average number of patients waiting in the different queues
    res.append(sum(map(len, env.queues)) / len(env.queues))    
    # adds waiting time in the queue in the feature along with the number of patients
    # wait time added should be adjusted when changing the number of steps in the model
    for q in env.queues:
        # Something about the length of the queue perhaps?
        res.append(np.mean([ p.need for p in q ]) if q else 0)
        res.append(np.mean([ p.wait for p in q ]) if q else 0)
    return np.array(res)

def feature_7(env):
    """A representation of the hospital
       A concatentation of:
       A one-hot vector for the need of the new patient
       A one-hot vector for the average wait
       One one-hot vector for the number of patients with a given need per queue
    """
    num_actions = len(env.actions)
    # Need of new patient
    res = np.array(np.arange(num_actions) == env.newPatient.need, dtype=int)
    # Average waiting time
    if sum(map(len, env.queues)) == 0:
        waits = 0
    else:
        # Not too sure about this one -- especially the boundary
        # waits = 1 if np.mean([ p.wait for p in np.concatenate(env.queues) ]) > num_actions*len(env.doctors) else 0
        waits = np.mean([ p.wait for p in np.concatenate(env.queues) ])
        if waits < num_actions: waits = 0
        elif waits < 2 * num_actions: waits = 1
        else: waits = 2
    waits = np.array(np.arange(3) == waits, dtype = int)
    # Number of people of given type in each queue
    types = np.array([])
    for i, q in enumerate(env.queues):
        q = [ p.need for p in q ]
        if q.count(i) < num_actions: c = 0
        elif q.count(i) < 2*num_actions: c = 1
        else: c = 2
        # types.extend( q.count(i) for i in range(num_actions) )
        # types = np.array( np.array(types) > num_actions, dtype = int )
        types = np.concatenate(( types, np.array(np.arange(3) == c, dtype = int)))
    return np.concatenate((res, waits, types))

def feature_8(env):
    """
    A representation of the hospital
       A concatentation of:
       A one-hot vector for the need of the new patient
       One one-hot vector for the number of patients with a given need per queue
       

       Difference with feature_7:
           
            ------- currently we do not include the wait vector in the feature representation
       
    """
    num_actions = len(env.actions)
    # Need of new patient
    res = np.array(np.arange(num_actions) == env.newPatient.need, dtype=int)
    # Number of people of given type in each queue
    types = np.array([])
    for i, q in enumerate(env.queues):
        q = [ p.need for p in q ]
        if q.count(i) < num_actions: c = 0
        elif q.count(i) < 2*num_actions: c = 1
        else: c = 2
        # types.extend( q.count(i) for i in range(num_actions) )
        # types = np.array( np.array(types) > num_actions, dtype = int )
        types = np.concatenate(( types, np.array(np.arange(3) == c, dtype = int)))
    return np.concatenate((res, types))


def feature_9(env):
    """A representation of the hospital
       A concatentation of:
       A one-hot vector for the total number of patients in queues
       A one-hot vector for the need of the new patient
       A one-hot vector for the average wait
       One one-hot vector for the number of patients with a given need per queue
       
       ----- difference with inital feature 7 -> added a vector for total number of patients
    """
    num_patients = sum(map(len, env.queues))
    if num_patients < 0.25 * env.occupancy: 
        num_patients = 0
    elif num_patients < 0.5 * env.occupancy:
        num_patients = 1
    elif num_patients < 0.75 * env.occupancy:
        num_patients = 2
    else:
        num_patients = 3
    num_patients = np.array(np.arange(4) == num_patients, dtype=int)

    num_actions = len(env.actions)
    # Need of new patient
    res = np.array(np.arange(num_actions) == env.newPatient.need, dtype=int)
    # Average waiting time
    if sum(map(len, env.queues)) == 0:
        waits = 0
    else:
        # Not too sure about this one -- especially the boundary
        # waits = 1 if np.mean([ p.wait for p in np.concatenate(env.queues) ]) > num_actions*len(env.doctors) else 0
        waits = np.mean([ p.wait for p in np.concatenate(env.queues) ])
        if waits < num_actions: waits = 0
        elif waits < 2 * num_actions: waits = 1
        else: waits = 2
    waits = np.array(np.arange(3) == waits, dtype = int)
    # Number of people of given type in each queue
    types = np.array([])
    for i, q in enumerate(env.queues):
        q = [ p.need for p in q ]
        if q.count(i) < num_actions: c = 0
        elif q.count(i) < 2*num_actions: c = 1
        else: c = 2
        # types.extend( q.count(i) for i in range(num_actions) )
        # types = np.array( np.array(types) > num_actions, dtype = int )
        types = np.concatenate(( types, np.array(np.arange(3) == c, dtype = int)))
    return np.concatenate((num_patients, res, waits, types))

def feature_10(env):
    """A representation of the hospital
       A concatentation of:
    
       One one-hot vector for the number of patients with a given need per queue
       
       A bad featurisation - does not carry enough information
    """
    num_actions = len(env.actions)
    # Need of new patient
    res = np.array(np.arange(num_actions) == env.newPatient.need, dtype=int)
    
    # Number of people of given type in each queue
    types = np.array([])
    for i, q in enumerate(env.queues):
        q = [ p.need for p in q ]
        if q.count(i) < num_actions: c = 0
        elif q.count(i) < 2*num_actions: c = 1
        else: c = 2
        # types.extend( q.count(i) for i in range(num_actions) )
        # types = np.array( np.array(types) > num_actions, dtype = int )
        types = np.concatenate(( types, np.array(np.arange(3) == c, dtype = int)))
    #return np.concatenate((res, types))
    return types

def feature_11(env):
    """A representation of the hospital
       A concatentation of:
       A one-hot vector for the need of the new patient
       A one-hot vector for the average wait
       One one-hot vector for the number of patients with a given need per queue
       
       ------ difference with feature_7 -> should have different definitions(boundaries)
       for the creation of the 1-hot vectors
    """
    num_actions = len(env.actions)
    h_capacity = env.occupancy
    # Need of new patient
    res = np.array(np.arange(num_actions) == env.newPatient.need, dtype=int)
    # Average waiting time
    if sum(map(len, env.queues)) == 0:
        waits = 0
    else:
        # Not too sure about this one -- especially the boundary
        # waits = 1 if np.mean([ p.wait for p in np.concatenate(env.queues) ]) > num_actions*len(env.doctors) else 0
        waits = np.mean([ p.wait for p in np.concatenate(env.queues) ])
        if waits < 0.1 * h_capacity: waits = 0
        elif waits < 0.3 * h_capacity: waits = 1
        else: waits = 2
    waits = np.array(np.arange(3) == waits, dtype = int)
    # Number of people of given type in each queue
    types = np.array([])
    for i, q in enumerate(env.queues):
        q = [ p.need for p in q ]
        if q.count(i) < 0.05 * h_capacity: c = 0
        elif q.count(i) < 0.1 * h_capacity: c = 1
        else: c = 2
        # types.extend( q.count(i) for i in range(num_actions) )
        # types = np.array( np.array(types) > num_actions, dtype = int )
        types = np.concatenate(( types, np.array(np.arange(3) == c, dtype = int)))
    return np.concatenate((res, waits, types))

def feature_12(env):
    """A representation of the hospital
       A concatentation of:
       A one-hot vector for the need of the new patient
       A one-hot vector for the average wait
       One one-hot vector for the number of patients with a given need per queue
       
       ------ difference with feature_7 -> should have info about the number of patients
       in each queue; also the boundaries
    """
    num_actions = len(env.actions)
    h_capacity = env.occupancy
    # Need of new patient
    res = np.array(np.arange(num_actions) == env.newPatient.need, dtype=int)
    
    # Number of people of given type in each queue and max waiting time
    types = np.array([])
    for i, q in enumerate(env.queues):
        if len(q) == 0:
            waits = 0
        else:
            waits = q[0].wait   # wait time of the first patient in line
            if waits < 0.1 * h_capacity: waits = 0
            elif waits < 0.3 * h_capacity: waits = 1
            else: waits = 2
        types = np.concatenate(( types, np.array(np.arange(3) == waits, dtype = int)))   
        #waits = np.array(np.arange(3) == waits, dtype = int)
        q = [ p.need for p in q ]
        if q.count(i) < 0.05 * h_capacity: c = 0
        elif q.count(i) < 0.1 * h_capacity: c = 1
        else: c = 2
        # types.extend( q.count(i) for i in range(num_actions) )
        # types = np.array( np.array(types) > num_actions, dtype = int )
        types = np.concatenate(( types, np.array(np.arange(3) == c, dtype = int)))
    return np.concatenate((res, types))

###### LEARNING ALGORITHMS ##################
def sarsa(env, featurisation, gamma, alpha, epsilon, num_episodes, num_steps, checkBefore = True, cap_penalty = False):
    """
      Currently returns a list with the rewards from each episode estimated using sarsa algorithm

      parameters
      ----------
      env - an environment that can be reset and interacted with via step
          (the hospital object in our case)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate, if defined as None then it is calculated as 1 / number_steps
      epsilon - the decision variable for how greedy the policy is. Epsilon = 1
                leads to a completely exploratory (random) action taking. Epsilon = 0
                is the fully greedy policy. 
      num_episodes - number of episode to run
      num_steps - number of steps per episode
      checkBefore - Whether misallocation is penalized at the beginning (default) or end of the queue
                    this is a parameter in the next_step function
      cap_penalty - Whether the agent is penalised when occupancy is reached

      returns
      -------
      Q_weights - the weight matrix, 1 weight vector for each action in the simulation
      total_reward_per_episode - a list with the reward for each episode
      t_list - List of when each episode terminated
    """
    if alpha == None:
        alpha = 1 / num_steps
    # list that saves the episodes that have been stopped prematurely
    t_list = []
    # used for the graphs at the end
    #total_reward_per_step = np.zeros(num_steps)
    #total_reward_per_episode = np.zeros(num_episodes)
    total_reward_per_episode = [-np.inf for _ in range(num_episodes)]
    #timeline_steps = [i for i in range(num_steps)]
    #timeline_episodes = [i for i in range(num_episodes)]
    
    # the number of possible doctor assignments for a patient
    num_actions = len(env.actions)
    
    # Q_weights is the weight matrix
    s = featurisation(env)
    Q_weights = np.ones((num_actions, len(s)))
    Q_optimal_weights = Q_weights
    
    #### trying to see if there is a best weight matrix
        
    # Apply Sarsa algorithm
    for episode in range(num_episodes):
        
        # variable to follow the reward evolution after each step
        reward = 0
        # reset the hospital object for the next episode
        env.reset()
        s = featurisation(env)
        a = epsilon_greedy(s, Q_weights, epsilon)
       
        for step in range(num_steps):
            # finish the current episode if the max occupancy is reached
            if env.isTerminal:
                t_list.append("Episode " + str(episode) + " finished at step " + str(step))
                break
            
            step_reward = env.next_step(a, checkBefore, cap_penalty)
            reward += step_reward
            s_ = featurisation(env)
            a_ = epsilon_greedy(s_, Q_weights, epsilon)
            
            Q_weights = sarsa_update(Q_weights, s, a, step_reward, s_, a_, gamma, alpha)
            
            s = s_
            a = a_
            # print("\nStep: {}\n".format(step))
            # env.pretty_print()
        
        # now add to the total reward from the episode to the list
        total_reward_per_episode[episode] = reward
        # print("\nEpisode: {}\n".format(step))
        
        ####################################
        # created a variable Q_optimal_weights 
        # i.e. this is is the weight matrix linked with the highest episodic reward
        calculate_Optimal_weights = True
        if calculate_Optimal_weights:
            if max(total_reward_per_episode) == total_reward_per_episode[episode]:
                #print("This is for epsilon ->{}".format(epsilon))
                Q_optimal_weights = Q_weights
                #print("Optimal weights changed in episode {}; reward -> {}\n".format(episode, total_reward_per_episode[episode]))
                if total_reward_per_episode[episode] > -500:
                    # used for testing
                    #env.pretty_print()
                    pass
        ####################################
        
    # return the final weight matrix and the list with episodic rewards
    return t_list, Q_optimal_weights, total_reward_per_episode

def sarsa_update(qweights, s, a, r, s_, a_, gamma, alpha):
    """
    Updates the qweights following the sarsa method for function approximation.
    ----------
    s    - is the 1d numpy array of the state feature
    a        - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    """
    q_current = q_approx(s, a, qweights)
    q_next = q_approx(s_, a_, qweights)
    qweights[a] += alpha*(r + gamma*q_next - q_current)*s
    return qweights

def ql(env, featurisation, gamma, alpha, epsilon, num_episodes, num_steps, checkBefore = True, cap_penalty = False):
    """
      Implementation of q-learning algorithm

      parameters
      ----------
      env - an environment that can be reset and interacted with via step
          (the hospital object in our case)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      num_steps - number of steps per episode

      returns
      -------
      Q_weights - the weight matrix, 1 weight vector for each action in the simulation
      total_reward_per_episode - a list with the reward for each episode
      t_list - List of when each episode terminated
    """
    if alpha == None:
        alpha = 1 / num_steps
    # list that saves the episodes that have been stopped prematurely
    t_list = []
    # used for the graphs at the end
    #total_reward_per_step = np.zeros(num_steps)
    total_reward_per_episode = np.zeros(num_episodes)
    #timeline_steps = [i for i in range(num_steps)]
    #timeline_episodes = [i for i in range(num_episodes)]
    
    # the number of possible doctor assignments for a patient
    num_actions = len(env.actions)
    
    # Q_weights is the weight matrix
    s = featurisation(env)
    Q_weights = np.zeros((num_actions, len(s)))
    Q_optimal_weights = Q_weights

    for episode in range(num_episodes):
        # variable to follow the reward evolution after each step
        reward = 0
        # reset the hospital object for the next episode
        env.reset()
        s = featurisation(env)
        a = epsilon_greedy(s, Q_weights, epsilon)
       
        for step in range(num_steps):
            # finish the current episode if the max occupancy is reached
            if env.isTerminal:
                t_list.append("Episode " + str(episode) + " finished at step " + str(step))
                break
            
            a = epsilon_greedy(s, Q_weights, epsilon)
            step_reward = env.next_step(a, checkBefore, cap_penalty)
            reward += step_reward
            s_ = featurisation(env)
            
            Q_weights = ql_update(Q_weights, s, a, step_reward, s_, gamma, alpha)
            
            s = s_
        # now add to the total reward from the episode to the list
        total_reward_per_episode[episode] = reward
        
        ####################################
        # created a variable Q_optimal_weights 
        # i.e. this is is the weight matrix linked with the highest episodic reward
        calculate_Optimal_weights = True
        if calculate_Optimal_weights:
            if max(total_reward_per_episode) == total_reward_per_episode[episode]:
                #print("This is for epsilon ->{}".format(epsilon))
                Q_optimal_weights = Q_weights
                #print("Optimal weights changed in episode {}; reward -> {}\n".format(episode, total_reward_per_episode[episode]))
        ####################################

    return t_list, Q_optimal_weights, total_reward_per_episode

def ql_update(qweights, s, a, r, s_, gamma, alpha):
    """
    Updates the qweights following the q-learning method for function approximation.
    Inputs
    ----------
    s    - is the 1d numpy array of the state feature
    a        - is the index of the action
    qweights - a list of weight vectors, one per action

    Returns
    -------
    Updated qweights
    """
    q_current = q_approx(s, a, qweights)
    q_next = np.argmax([ q_approx(s_, a_, qweights) for a_ in range(len(qweights)) ])
    qweights[a] += alpha*(r +gamma*q_next - q_current)*s
    return qweights

###########################################################
# the functions from example.py are below                 #
###########################################################


# Value-function approximation
def q_approx(s, a, qweights):
    """
    parameters
    ----------
    s - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    the q_value approximation for the state and action input
    """
    method = "Dot product"
    
    if method == "Dot product":
        return np.dot(s, qweights[a])
    elif method == "Fourier":
        pass

##### POLICIES ################################  
def epsilon_greedy(s, qweights, epsilon):
    """
    A method to sample from the epsilon greedy policy associated with a
    set of q_weights which captures a linear state-action value-function

    parameters
    ----------
    s - is the 1d numpy array of the state feature
    epsilon - probability of acting randomly
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    decision taken using the epsilon-greedy policy 
    """
    if np.random.random() > epsilon:
        return np.argmax(
            [ q_approx(s, a, qweights) for a in range(len(qweights)) ])
    return np.random.randint(len(qweights))

# Can I formulate this in terms of weights?
def policy_naive(env):
    """Allocate the patient to the queue that corresponds to their need"""
    return env.newPatient.need

##### Visualisations ###########################
def simulate(
    env,
    featurisation,
    q_weights,
    steps = 100,
    epsilon = 0,
    plot = False,
    title = "Untitled",
    printSteps = 0,
    checkBefore = True,
    cap_penalty = False):
    """ 
    Simulates a hospital using the epsilon-greedy
    policy based on weights and can plot a stacked bar plot with the results

    Inputs:
    -----------
    env - a hospital instance
    steps - how many steps to simulate
    featurisation - outputs a representation for a given hospital state
    q_weights - a set of weights which was learned;
        these should be based on learning done using the above featurisation
    epsilon - probability of choosing action randomly
    plot - What kind of plot (if any) should be produced: 
        "hist" or "map" (stacked bar plot or heatmap)
    title - If there is a plot, it's title (string)
    printSteps - If non-zero, this will print the internal state of the hospital every printSteps steps
    checkBefore - During training, whether to penalise the agent for misallocation as soon as it's done or once the patient gets to the doctor
    cap_penalty - During training, whether to penalise the agent when the hospital capacity is reached (and the episode is terminated)

    """
    env.reset()
    N = len(env.actions)
    props = np.zeros([N, N])
    rewards = np.zeros(steps)

    for i in range(steps):
        s = featurisation(env)
        a = epsilon_greedy(s, q_weights, epsilon)
        props[env.newPatient.need, a] += 1
        rewards[i] = env.next_step(a, checkBefore, cap_penalty)
        if printSteps and not(i%printSteps):
        # Only print if printSteps is true and the step is a multiple of printSteps
            print("Reward:", rewards[i])
            env.pretty_print()

    props = props/steps*100
    
    # Would be nice to give some kind of list as input to give all the plots, not just one
    if plot == "hist":
        fig, ax = plt.subplots()
        cumsum = props[0,:]
        p = [ax.bar(range(N), cumsum, color=(172/255,255/255,47/255))]
        for i in range(1,N):
            p.append(ax.bar(range(N), props[i,:], bottom = cumsum))
            # Tried to get expressive colors but it makes them more difficult to distinguish
            # p.append(ax.bar(range(N), props[i,:], bottom = cumsum, color = ( (i+1)/N, 1-(i+1)/N, 0 )))
            cumsum = cumsum + props[i,:]

        plt.ylabel("Proportion of each patient type by queue")
        plt.title(title)
        plt.xticks(range(N), ['Queue '+str(i) for i in range(N)] )
        plt.yticks(np.arange(0, 101, 10))
        plt.legend(tuple( p[i][0] for i in range(N) ), tuple('Type '+str(i) for i in range(N)))

    elif plot == "map":
        fig, ax = plt.subplots()
        im = ax.imshow(props.T, cmap=plt.get_cmap("RdYlGn").reversed())
        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        # ... and label them with the respective list entries
        ax.set_yticklabels(["Queue "+str(i) for i in range(N)])
        ax.set_xticklabels(range(N))

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        cbar = ax.figure.colorbar(im, ax=ax)
        
        ax.set_title("Proportion of each patient type by queue")
        ax.set_title(title)
        fig.tight_layout()

    # some extra metrics to estimate the performance of the algorithm
    # number of cured patients
    n_cured = len(env.cured)
    
    # will show number of people cured by type
    cured_dict = dict()  
    cured_types = [patient.need for patient in env.cured]    
    
    for i in range(len(env.queues)):
        cured_dict["Type " + str(i)] = cured_types.count(i)

    # combined waiting time for the cured patients
    time_waited_total = sum(patient.wait for patient in env.cured)
    time_array = [ [ p.wait for p in filter(lambda x: x.need == i, env.cured) ] for i in range(len(env.actions)) ]

    plt.show()
    return props, rewards, n_cured, time_array, cured_dict


def simulate_naive(env, steps = 100, plot = False, checkBefore = True, cap_penalty = False):
    """ 
    Simulates a hospital using the naive policy

    Inputs:
    -----------
    env: a hospital instance
    steps: how many steps to simulate
    plot: What kind of plot (if any) should be produced: "hist" or "map"
        (stacked bar plot or heatmap)
    """
    env.reset()
    N = len(env.actions)
    props = np.zeros([N, N])
    rewards = np.zeros(steps)
    for i in range(steps):
        # s = featurisation(env)
        a = policy_naive(env)
        props[env.newPatient.need, a] += 1
        rewards[i] = env.next_step(a, checkBefore, cap_penalty)

    props = props/steps*100
    
    if plot == "hist":
        fig, ax = plt.subplots()
        cumsum = props[0,:]
        p = [ax.bar(range(N), cumsum, color=(172/255,255/255,47/255))]
        for i in range(1,N):
            p.append(ax.bar(range(N), props[i,:], bottom = cumsum))
            # p.append(ax.bar(range(N), props[i,:], bottom = cumsum, color = ( (i+1)/N, 1-(i+1)/N, 0 )))
            cumsum = cumsum + props[i,:]

        plt.ylabel("Proportions")
        plt.title('Proportion of patients in queues - naive policy')
        plt.xticks(range(N), ['Queue '+str(i) for i in range(N)] )
        plt.yticks(np.arange(0, 101, 10))
        plt.legend(tuple( p[i][0] for i in range(N) ), tuple('Type '+str(i) for i in range(N)))
        fig.tight_layout()

    elif plot == "map":
        fig, ax = plt.subplots()
        im = ax.imshow(props.T, cmap=plt.get_cmap("RdYlGn").reversed())
        ax.set_xticks(np.arange(N))
        ax.set_yticks(np.arange(N))
        # ... and label them with the respective list entries
        ax.set_yticklabels(["Queue "+str(i) for i in range(N)])
        ax.set_xticklabels(range(N))

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        # for i in range(len(vegetables)):
            # for j in range(len(farmers)):
                # text = ax.text(j, i, harvest[i, j],
                       # ha="center", va="center", color="w")
                       
        cbar = ax.figure.colorbar(im, ax=ax)
        
        ax.set_title("Proportion of patients in queues - naive policy")
        fig.tight_layout()

    plt.show()

    return props, rewards

def rewards_curve(max_rewards, rand_rewards, sim_rewards, num_episodes, title = "Untitled", i = None, legend = True):
    # if fig is None:
    #     fig, ax = plt.subplots(tight_layout = True)
    # else:
    #     ax = fig.subplots()
    plt.title(title)
    # ax.plot(range(num_episodes), sim_rewards, "-b", figure = fig, label = "Learned policy")
    # ax.plot(range(num_episodes), max_rewards, "-g", figure = fig, label = "$r=0$ line")
    # ax.plot(range(num_episodes), rand_rewards, "-r", figure = fig, label = "Random policy")
    plt.plot(range(num_episodes), sim_rewards, "-b", label = "Learned policy")
    #plt.plot(range(num_episodes), max_rewards, "-g", label = "$r=0$ line")
    plt.plot(range(num_episodes), rand_rewards, "-r", label = "Random policy")
    if legend:
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

    # return ax
