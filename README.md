# RL-hospital

Simulate a hospital and explore different reinforcement learning algorithms to optimize policy.

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

### Files
- `hospital.py` defines the `Hospital` class along with the methods necessary to simulate the passage of time in a hospital instance (including the reward system necessary for learning a policy).
- `learning.py` defines all methods pertaining to learning a policy: featurisations, learning algorithms, visualisation of policies.
- `simulation.py` contains various experiments, i.e. comparisons of learned policies in different situations, having only changed a few parameters.

## TODO
### On hold
- Look for different q-function approximations (RBFs? Monomials)? (Todor)
### Current
- Better document the code
- Experiments

## Report Plan
1. Intro:
- Background
- Describe the system
- Discuss details (e.g. when the rewards are applied)
- Introduce research (which algorithms, which experiments, what was compared)
- Introduce challenges (rewards given at the end)
2. Results
- Experiment 1: (Todor)
  - Fixed reward system
  - Compare featurisations:
    - One non-one-hot
    - `feature_7`
    - Another one-hot
- Experiment 2: (Auguste)
  - 4 doctor types
  - Change efficiency of most skilled doctor
  - Plot something vs. efficiency
- Experiment 3: (Isabel and Yongxin)
  - Fixed system
  - Fixed featurisation
  - Change reward system
3. Discussion
- Which featurisations do best;
  - Adapt it to the rewards system (e.g. if part of the reward depends on patient waits then you should include info about those in the featurisation)
- Which algorithms do best? -> ql seems faster? (to test)
- In what situations we can expect to find the optimal policy
- What challenges were there? How did we solve/not solve them?
- Global comparison/applicability
4. Conclusion/Opening
- How did it go compared to expectations?
- What could be done better?
  - Further function approximation? (RBFs, monomials)
  - "Marking scheme" (dividing reward into several parts)
