# RL-hospital

Simulate a hospital and explore different reinforcement learning algorithms to optimize policy.

## Files
- `hospital.py` defines the `Hospital` class along with the methods necessary to simulate the passage of time in a hospital instance (including the reward system necessary for learning a policy).
- `learning.py` defines all methods pertaining to learning a policy: featurisations, learning algorithms, visualisation of policies.
- `feat_exp.py` generates (or reads pre-generated) and plots data for experiment 1 of the report, where several featurisations are compared.
- `fastdoc_exp.py` generates (or reads pre-generated) and plots data for experiment 2 of the report, where we study the behaviour of the agent 
- `rewards_exp.py` generates (or reads pre-generated) and plots data for experiment 3 of the report, where several algorithms and reward systems are compared. 
- `data_combine.py` is used to combine data arrays from different files: that way, all the data doesn't have to come from a single run of the experiment script.
- `exp1`, `exp2` and `exp3` contain generated data for each experiment. When an experiment is run in "generate" mode, it automatically outputs the generated data to the corresponding directory.
- (deprecated )`misalloc_check.py` generates data for a plot that used to be in experiment 3

Each experiment can be run in "generate" or "read" mode; if the file is run with an argument, it assumes a data file in numpy binary format, reads the data from it and plots this data. Otherwise (no arguments), it generates new data according to the number of trials, steps and episodes given in the file, outputs the data to a .npz file for later use and plots it.

## Description
The hospital has different doctors who specialize in different things.

For example, doctors could be of 3 different types: type 0, type 1 and type 2.
A higher number corresponds to higher experience.
Each individual doctor also has a certain efficiency (modelled as a probability of being finished treating a patient).

Each *type of doctor* has an associated queue, and there can be any number of doctors for each type.

When patients arrive at the hospital, they shall be treated according to their needs.

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
For example, a patient with need 0 can be treated by any doctor, but a patient with need 2 can't be treated by a doctor of type 0 or of type 1, and will be sent away *but only once the patient has gone through the queue*.
