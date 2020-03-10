# RL-hospital

Simulate a hospital and explore different reinforcement learning algorithms to optimize policy.


The hospital has different doctors who specialize in different things.
For example, doctors can be of 3 different types: "Non-urgent", "Urgent" and "Critical".
Each individual doctor has an associated queue, and there can be any number of doctors for each type.

When patients arrive at the hospital, they shall be treated according to their needs.
Specifically, a patient with "Critical" needs has to be seen by a "Critical" doctor, yet a patient with "Non-urgent" needs can be seen by any kind of doctor.

The agent dispatches arriving patients to some queue, where they wait to be seen.

The total number of patients in the hospital at any given time is bounded.
