# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:18:12 2020

@author: todor

This code has been adapted from code provided by Luke Dickens on the UCL module INST0060: Foundations of Machine Learning and Data Science
"""


import numpy as np
import random

##### FEATURISATIONS ##########################
def feature(env):
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
            elif num_patients <= 2:
                res.append(1)
            else:
                res.append(2)
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
    """
    
    # list that saves the episodes that have been stopped prematurely
    t_list = []
    # used for the graphs at the end
    #total_reward_per_step = np.zeros(num_steps)
    total_reward_per_episode = np.zeros(num_episodes)
    #timeline_steps = [i for i in range(num_steps)]
    timeline_episodes = [i for i in range(num_episodes)]
    
    # the number of possible doctor assignments for a patient
    num_actions = len(env.actions)
    
    # Q_weights is going to be the weight matrix
    s = featurisation(env)
    Q_weights = np.zeros((num_actions, len(s)))
        
    # now we simulate the Sarsa algorithm
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
                t_list.append("t_episode:" + str(episode) + " step:" + str(step) + "\n")
                
                break
            
            step_reward = env.next_step(a)
            reward += step_reward
            s_ = featurisation(env)
            a_ = sample_from_epsilon_greedy(s_, Q_weights, epsilon)
            
            Q_weights = sarsa_update(Q_weights, s, a,
                                     step_reward, s_, a_,
                                     gamma, alpha)
            
            s = s_
            a = a_
            print("\nStep: {}\n".format(step))
            env.pretty_print()
        # now add to the total reward from the episode to the list
        total_reward_per_episode[episode] = reward
       # print("\nStep: {}\n".format(step))
        
            
    # return the final weight matrix and the list with episodic rewards
    return t_list, Q_weights, total_reward_per_episode, timeline_episodes


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
def sample_from_epsilon_greedy(s_rep, qweights, epsilon):
    """
    A method to sample from the epsilon greedy policy associated with a
    set of q_weights which captures a linear state-action value-function

    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    """
    qvalues = np.empty(qweights.shape[0])
    for a in range(qweights.shape[0]):
        qvalues[a] = state_action_value_function_approx(s_rep, a, qweights)
    if np.random.random() > epsilon:
      return np.argmax(qvalues)
    return np.random.randint(qvalues.shape[0])

def policy_random(qweights):
    """Sample from the random policy"""
    # Return one of the possible actions with probability 1/(number of possible actions)
    return random.choice(range(qweights.shape[0]))
