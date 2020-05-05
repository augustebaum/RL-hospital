from hospital import *
from learning import *
import numpy as np
import matplotlib.pyplot as plt

# For detecting the extra arguments
import sys
import os

# For naming exported files automatically
from datetime import datetime

""" 
You can run this file as is:
``` python3 fastdoc_exp.py ```
which will generate data and plot it.
Depending on the training parameters (number_episodes, number_steps),
this may take time.
If all you care about is seeing the data used in the report (or any previously generated data),
you can pass a file as an argument:
``` python3 fastdoc_exp.py exp2/data.npz ```
This will load the data contained in data.npz and plot it instead of generating new data.

TODO:
    Group data arrays into multidimensional array
    Allow several files as input (plot data in each of them)
"""

# Careful, all the data was generated using these probabilities so changing them will make it impossible to plot pre-exisiting data
p_array = np.linspace(0.1, 0.9, 5)

feature = feature_12  # One-hot
capacity_hospital = 500
number_steps = 100
number_episodes = 30
number_tries = 5
algorithm = sarsa


def main(p_array=p_array, number_tries=5):
    if len(sys.argv) == 1:
        # If no argument
        data = generate(p_array, number_tries)
    else:  # Assume only 1 extra argument: npz file containing array
        dict = np.load(str(sys.argv[1]))
        data = tuple(map(lambda x: x[1], dict.items()))
        dict.close()

    plot(*data, p_array)


def generate(p_array, number_tries):
    """
    Generates data for experiment 2

    Inputs:
    p_array      - array of probabilities (type 3 arrivals)
    number_tries - Number of trials per probability

    Outputs:
    data arrays with len(p_array) lines and number_tries columns 
    """
    # Number of people in queue 3 for each p
    # Number of people in queue 3 that are of type 3 for each p
    # Average amount of time type 3 patients waited for
    (queue3, queue3type3, time3) = tuple(
        np.empty([number_tries, len(p_array)])\
        for _ in range(3)
    )

    ###############################################

    # Efficiency of low skill doctors
    p_slow = 0.4
    # Efficiency of high skill doctor
    p_fast = 0.8

    doctors = [
        Doctor(0, p_slow),
        Doctor(1, p_slow),
        Doctor(2, p_slow),
        Doctor(3, p_fast),
    ]

    ##############################################

    for j, p in enumerate(p_array):
        arrival_rates = [1, 1, 1, round(3 * p / (1 - 0.99999 * p))]

        # Train, simulate and gather:
        # - The number of patients in queue 3
        # - The number of patients of type 3 in queue 3
        # - The time that 3-patients waited

        for i in range(number_tries):
            # Testing using equal arrival probabilities to provide unbiased account
            props, _, _, time_array, _, _ = test(
                algorithm,
                capacity_hospital,
                number_steps,
                number_episodes,
                arrival_rates,
                doctors,
                feature,
                p_prob_test=[1, 1, 1, 1],
                gamma=0.9,
                alpha=None,
                epsilon=0.1,
                plot_type=None,
                title1="Type 3 patients arrive {:.0%} of the time during training".format( p ),
                title2="Reward evolution for the picture above",
            )

            queue3[i, j] = sum(props[:, 3])
            queue3type3[i, j] = props[3, 3]
            time3[i, j] = np.mean(time_array[3]) if time_array[3] else 0

    # Save to file for further analysis (deactivated)
    # np.savez(
    #     os.path.dirname(os.path.realpath(__file__))
    #     + "/exp2/exp2--"
    #     + str(number_episodes)
    #     + "episodes"
    #     + str(number_steps)
    #     + "steps"
    #     + datetime.now().strftime("%H-%M-%S"),
    #     queue3,
    #     queue3type3,
    #     time3,
    # )

    return queue3, queue3type3, time3


def errors(arr):
    """
    Distance from median to 20th and 80th percentile, for error bars
    """
    a = np.median(arr, axis=0)
    return a, np.abs(np.quantile(arr, [0.20, 0.80], axis=0) - a)


def plot_errorbar(points_array, label):
    plt.errorbar(
        p_array,
        points_array[0],
        yerr=points_array[1],
        elinewidth=1,
        capsize=2,
        label=label,
    )


def plot(queue3, queue3type3, time3, p_array):
    # Could be made more extensible with multidimensional array
    plt.figure(1)
    plot_errorbar(errors(queue3), "Proportion of patients in queue 3")
    plot_errorbar(errors(queue3type3), "Proportion of patients of type 3 in queue 3")
    plt.axis([0, 1, -1, 101])
    plt.legend()
    plt.xlabel("Probability that arriving patient has type 3 during training")
    plt.tight_layout()

    plt.figure(2)
    plot_errorbar(errors(time3), "Average time waited by (cured) type 3 patients")
    plt.axis([0, 1, -0.05, plt.axis()[3] + 0.05])
    plt.legend()
    plt.xlabel("Probability that arriving patient has type 3 during training")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main(number_tries=5)
