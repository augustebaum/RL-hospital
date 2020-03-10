# RL-hospital

Simulate a hospital and explore different reinforcement learning algorithms to optimize policy.

## Description
The hospital has different doctors who specialize in different things.

For example, doctors could be of 3 different types: "Non-urgent", "Urgent" and "Critical".
Each individual doctor has an associated queue, and there can be any number of doctors for each type.

When patients arrive at the hospital, they shall be treated according to their needs.
Specifically, a patient with "Critical" needs **has to be seen** by a "Critical" doctor, yet a patient with "Non-urgent" needs **can be seen by any kind of doctor**.

## Modelling

This is a discrete-time problem. At each time step:
- One or more patients can arrive at the hospital, and state the kind of needs they have. The agent immediately dispatches them to some queue, where they wait to be seen.
- One or more doctors can finish treating patients, in which case they immediately start treating the first person in their queue.

The state is a list of lists, each list representing a queue. In each queue, patients are represented by an integer which corresponds to the number of timesteps they have spent in the queue.

The total number of patients in the hospital at any given time is bounded.
When the hospital is full, any arriving patient is sent away (with a penalty for the agent).
