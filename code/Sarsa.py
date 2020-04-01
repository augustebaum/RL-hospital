# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:18:12 2020

@author: todor

This code has been adapted from code provided by Luke Dickens on the UCL module INST0060: Foundations of Machine Learning and Data Science
"""


import numpy as np


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



def sarsa(env, gamma, alpha, epsilon, num_episodes, num_steps):
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
    
    # used for the graphs at the end
    #total_reward_per_step = np.zeros(num_steps)
    total_reward_per_episode = np.zeros(num_episodes)
    #timeline_steps = [i for i in range(num_steps)]
    timeline_episodes = [i for i in range(num_episodes)]
    
    # the number of possible doctor assignments for a patient
    num_actions = len(env.actions)
    
    # Q_weights is going to be the weight matrix
    feature_rep = feature_1(env)
    Q_weights = np.zeros((num_actions, len(feature_rep)))
        
        
        
    # now we simulate the Sarsa algorithm
    for episode in range(num_episodes):
        
        # variable to follow the reward evolution after each step
        reward = 0
        # reset the hospital object for the next episode
        env.reset()
        
        current_state_feature = feature_1(env)
        current_action = sample_from_epsilon_greedy(current_state_feature,
                                                    Q_weights, epsilon)
        
        for step in range(num_steps):
            
            step_reward = env.next_step(current_action)
            reward += step_reward
            next_state_feature = feature_1(env)
            next_action = sample_from_epsilon_greedy(next_state_feature,
                                                    Q_weights, epsilon)
            
            Q_weights = sarsa_update(Q_weights, current_state_feature, current_action,
                                     step_reward, next_state_feature, next_action,
                                     gamma, alpha)
            
            current_state_feature = next_state_feature
            current_action = next_action
            
        # now add to the total reward from the episode to the list
        total_reward_per_episode[episode] = reward
            
            
    # return the final weight matrix and the list with episodic rewards
    return Q_weights, total_reward_per_episode, timeline_episodes



###########################################################
# the functions from example.py are below                 #
###########################################################



def state_action_value_function_approx(s_rep, a, qweights):
    """
    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    the q_value approximation for the state and action input
    """
    qweights_a = qweights[a]
    return np.dot(s_rep, qweights_a)

def sarsa_update(qweights, s_rep, a, r, next_s_rep, next_a, gamma, alpha):
    """
    A method that updates the qweights following the sarsa method for
    function approximation. You will need to integrate this with the full
    sarsa algorithm
    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    """
    q_current = state_action_value_function_approx(s_rep, a, qweights)
    q_next = state_action_value_function_approx(next_s_rep, next_a, qweights)
    DeltaW = alpha*(r +gamma*q_next - q_current)*s_rep
    qweights[a] += DeltaW
    return qweights

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




