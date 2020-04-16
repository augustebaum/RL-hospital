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
- Look for different q-function approximations (RBFs? Monomials)? (Todor)
### Current
- Find more featurisations (Todor)
- Merge branches, add possibility to have rewards at the beginning or end in next\_step (Auguste)
- Work on hospital with 2 doctors (fixed featurisation and rewards, change rates) (Isabel and Yongxin)
- Report:
  - Shorten introduction (Auguste)
  - Describe experiments (what is being compared, what is expected)
    - Experiment 1 (Todor)

## Report Plan
1.
Intro:
- Background
- Describe the system
- Discuss details (e.g.
when the rewards are applied)
- Introduce research (which algorithms, which experiments, what was compared)
- Introduce challenges (rewards given at the end)
2.
Results
- Experiment 1:
 - Fixed featurisation
 - Fixed reward system
 - Doctors = [0, 1]
 - Compare effect of having more or less type 1 patients
- Experiment 2:
 - 7 doctor types
 - Compare featurisation (one-hot and normal)
3.
Discussion
- Which featurisations do best;
- Which algorithms do best?
- In what situations we can expect to find the optimal policy
- What challenges were there? How did we solve/not solve them?
- Global comparison/applicability
4.
Conclusion/Opening
- How did it go compared to expectations?
- What could be done better?
  - Further function approximation? (RBFs, monomials)
  - "Marking scheme" (dividing reward into several parts)
