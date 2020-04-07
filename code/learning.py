# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:18:12 2020

@author: todor

This code has been adapted from code provided by Luke Dickens on the UCL module INST0060: Foundations of Machine Learning and Data Science
"""
import matplotlib.pyplot as plt
import numpy as np
import random

##### FEATURISATIONS ##########################
def feature_1(env):
    """A representation of the hospital"""
    res = []
    # Average number of patients waiting in the different queues
    res.append(sum(map(len, env.queues))/len(env.queues))    
    # Whether or not a given queue has more or fewer patients with a certain need than a threshold
    for q in env.queues:
        q = [ p.need for p in q ]             
        for i in env.actions:
            num_patients = q.count(i)
            if num_patients < 1:
                res.append(0)
            elif num_patients <= 1:
                res.append(1)
            elif num_patients <= 3:
                res.append(2)
            else:
                res.append(3)
    return np.array(res)

def feature_2(env):
    """A representation of the hospital"""
    res = []
    # Average number of patients waiting in the different queues
    res.append(sum(map(len, env.queues))/len(env.queues))    
    # Whether or not a given queue has more or fewer patients with a certain need than a threshold
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
        wait_time = sum([p.wait for p in q])/100
        res.append(wait_time)
        for i in env.actions:
            res.append(q_.count(i))
    return np.array(res)

###### LEARNING ALGORITHMS ##################
def sarsa(env, featurisation, gamma, alpha, epsilon, num_episodes, num_steps):
    """
      Currently returns a list with the rewards from each episode estimated using sarsa algorithm

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
      timeline_episodes - just a list of the form [0,1,2,3,4.....,num_episodes] 
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
        
    # Apply Sarsa algorithm
    for episode in range(num_episodes):
        
        # variable to follow the reward evolution after each step
        reward = 0
        # reset the hospital object for the next episode
        env.reset()
        s = featurisation(env)
        a = sample_from_epsilon_greedy(s, Q_weights, epsilon)
       
        for step in range(num_steps):
            # finish the current episode if the max occupancy is reached
            if env.isTerminal:
                t_list.append("Episode " + str(episode) + " finished at step " + str(step))
                break
            
            step_reward = env.next_step(a)
            reward += step_reward
            s_ = featurisation(env)
            a_ = sample_from_epsilon_greedy(s_, Q_weights, epsilon)
            
            Q_weights = sarsa_update(Q_weights, s, a, step_reward, s_, a_, gamma, alpha)
            
            s = s_
            a = a_
            # print("\nStep: {}\n".format(step))
            # env.pretty_print()
        # now add to the total reward from the episode to the list
        total_reward_per_episode[episode] = reward
       # print("\nEpisode: {}\n".format(step))
        
    # return the final weight matrix and the list with episodic rewards
    return t_list, Q_weights, total_reward_per_episode


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
    q_current = state_action_value_function_approx(s, a, qweights)
    q_next = state_action_value_function_approx(s_, a_, qweights)
    DeltaW = alpha*(r +gamma*q_next - q_current)*s
    qweights[a] += DeltaW
    return qweights


###########################################################
# the functions from example.py are below                 #
###########################################################


# Value-function approximation
def state_action_value_function_approx(s, a, qweights):
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
    return np.dot(s, qweights[a])

##### POLICIES ################################  
def sample_from_epsilon_greedy(s, qweights, epsilon):
    """
    A method to sample from the epsilon greedy policy associated with a
    set of q_weights which captures a linear state-action value-function

    parameters
    ----------
    s - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    decision taken using the epsilon-greedy policy 
    """
    qvalues = np.empty(qweights.shape[0])
    for a in range(qweights.shape[0]):
        qvalues[a] = state_action_value_function_approx(s, a, qweights)
    if np.random.random() > epsilon:
      return np.argmax(qvalues)
    return np.random.randint(qvalues.shape[0])

def policy_random(qweights):
    """Sample from the random policy"""
    # Return one of the possible actions with probability 1/(number of possible actions)
    return random.choice(range(qweights.shape[0]))

##### Visualisations ###########################
def simulate(env, featurisation, q_weights, steps = 100, epsilon = 0.1, plot = False):
    """ 
    Simulates a hospital using the epsilon-greedy
    policy based on weights and can plot a stacked bar plot with the results

    Inputs:
    -----------
    env: a hospital instance
    steps: how many steps to simulate
    featurisation: outputs a representation for a given hospital state
    q_weights: a set of weights which was learned; these should be based
        on learning done using the above featurisation
    epsilon: probability of choosing action randomly
    """
    N = len(env.actions)
    props = np.zeros([N, N])
    for _ in range(steps):
        s = featurisation(env)
        a = sample_from_epsilon_greedy(s, q_weights, epsilon)
        props[env.newPatient.need, a] += 1
        env.next_step(a)
    
    props = props/steps*100
    
    if plot:
        fig, ax = plt.subplots()
        cumsum = props[0,:]
        p = [ax.bar(range(N), cumsum, color=(0,1,0))]
        for i in range(1,N):
            p.append(ax.bar(range(N), props[i,:], bottom = cumsum, color = ( (i+1)/N, 1-(i+1)/N, 0 )))
            cumsum = cumsum + props[i,:]

        plt.ylabel("Proportions")
        plt.title('Proportion of each patient type by queue')
        plt.xticks(range(N), ['Queue '+str(i) for i in range(N)] )
        plt.yticks(np.arange(0, 101, 10))
        plt.legend(tuple( p[i][0] for i in range(N) ), tuple('Type '+str(i) for i in range(N)))

        plt.show()

    return props

    



