# RL-hospital

Simulate a hospital and explore different reinforcement learning algorithms to optimize policy.

## Description
The hospital has different doctors who specialize in different things.

For example, doctors could be of 3 different types: type 0, type 1 and type 2.
A higher number corresponds to higher experience.

[Each individual doctor has an associated queue, and there can be any number of doctors for each type.]::
Each *type of doctor* has an associated queue, and there can be any number of doctors for each type.

When patients arrive at the hospital, they shall be treated according to their needs.

The hospital doesn't provide appointment services (might be a walk-in centre).

## Modelling

This is modelled as a discrete-time process.

At each timestep:
- One patient arrives at the hospital, and declare the severity of their ailment (in the form of an `Int`).
The agent immediately dispatches them to some queue, where they wait to be seen.
- One or more doctors can finish treating patients, in which case they immediately start treating the first person in the queue.

The state is composed of:
- the queues for each type of doctor (e.g.
if there are doctors of type 0 and 1 then there will be 2 queues: one to see a doctor of type 0 and the other to see a doctor of type 1);
- doctors, each with their own type (a whole number) and efficiency (probability of being done at each timestep);
- a new patient with their characteristics (namely their need; a whole number, like patient types).
Patients have a waiting time (the number of timesteps they have spent in the queue).

To efficiently model the problem, classes `Doctor` and `Patient` have been implemented.
The queues contain `Patient` instances.

### Constraints
The total number of patients in the hospital is bounded.
When the hospital is full, any arriving patient is sent away.

A patient with need *n* has to be seen by a doctor of type *n* or above.
For example, a patient with need 0 can be treated by any doctor, but a patient with need 2 can't be treated by a doctor of type 0 or of type 1, and will be sent away *only once the patient has gone through the queue*.

## TODO
### On hold
- Implement some learning algorithms, *preferrably with reason* (Todor)
### Current
- Find one or more featurisations to try out (Use log) (Auguste) 
- Find a way to measure the model's performance (Auguste)
 -> Compare how the learner behaves as parameters are changed (e.g. doctor of higher type finishes faster -> does the learner give them more low priority patients?)
- Troubleshoot using simple cases
- Report (Isabel, Auguste, Yongxin)
#### Technical
- Look for different q-function approximations (RBFs? Monomials)? (Todor)
