# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:18:12 2020

@author: todor
"""


import numpy as np



def sarsa(env, gamma, alpha, epsilon, num_episodes, num_steps):
    """
      Estimates optimal policy by interacting with an environment using
      a td-learning approach

      parameters
      ----------
      env - an environment that can be reset and interacted with via step
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      policy - an estimate for the optimal policy
      Q - a Q-function estimate of the output policy
    """
    
    # used for the graphs at the end
    #total_reward_per_step = np.zeros(num_steps)
    total_reward_per_episode = np.zeros(num_episodes)
    #timeline_steps = [i for i in range(num_steps)]
    timeline_episodes = [i for i in range(num_episodes)]
    
    # the number of possible doctor assignments for a patient
    num_actions = len(env.actions)
    
    # Q_weights is going to be the weight matrix
    # currently corresponding to our initial featurisation
    Q_weights = np.zeros((num_actions, num_actions**2 + 1))
        
        
        
    # now we simulate the Sarsa algorithm
    for episode in range(num_episodes):
        
        # variable to follow the reward evolution after each step
        reward = 0
        # reset the hospital object for the next episode
        env.reset()
        
        current_state_feature = env.feature()
        current_action = sample_from_epsilon_greedy(current_state_feature,
                                                    Q_weights, epsilon)
        
        for step in range(num_steps):
            
            step_reward = env.next_step(current_action)
            reward += step_reward
            next_state_feature = env.feature() 
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




