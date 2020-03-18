# RL-hospital

Simulate a hospital and explore different reinforcement learning algorithms to optimize policy.

## Description
The hospital has different doctors who specialize in different things.

For example, doctors could be of 3 different types: "Non-urgent", "Urgent" and "Critical".

[Each individual doctor has an associated queue, and there can be any number of doctors for each type.]::
Each *type of doctor* has an associated queue, and there can be any number of doctors for each type.

When patients arrive at the hospital, they shall be treated according to their needs.
Specifically, a patient with "Critical" needs **has to be seen** by a "Critical" doctor, yet a patient with "Non-urgent" needs **can be seen by any kind of doctor**.

The hospital doesn't provide appointment services (might be a walk-in centre).

## Modelling

This is modelled as a discrete-time problem.

At each time step:
- One patient (no more) may arrive at the hospital, and declare the severity of their ailment (in the form of an Int).
The agent immediately dispatches them to some queue, where they wait to be seen.
- One or more doctors can finish treating patients, in which case they immediately start treating the first person in their queue.

The state is a list of Doctor instances, each with their own queue, along with a couple representing whether there is a patient waiting to be dispatched and if so, its need.
In each queue, patients are represented by an integer which corresponds to the number of timesteps they have spent in the queue.

The total number of patients in the hospital at any given time is bounded.
When the hospital is full, any arriving patient is sent away (with a penalty for the agent).

## TODO
- One patient arrives at every timestep;
- There is one queue per type of doctor;
- The reward system is embedded in the process itself (the `time-advance` function);
- A new, simpler representation should be implemented in order to evaluate and update the policy.
