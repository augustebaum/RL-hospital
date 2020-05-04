from hospital import *
from learning import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


""" Bar plot code (including `autolabel` function) obtained from:https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
"""

conditions = [(True, True), (True, False), (False, True), (False, False)]


def main(number_tries=5, number_episodes=100, number_steps=100):
    if len(sys.argv) == 1:
        data = generate(number_tries, number_steps, number_episodes)
    else:  # Assume only 1 extra argument: npz file containing array
        dict = np.load(str(sys.argv[1]))
        data = tuple(map(lambda x: x[1], dict.items()))
        dict.close()

    # Just in case there are NaN
    data = map(lambda arr: arr[~np.isnan(arr).any(axis=1)], data)

    plot(*data)


def generate(n_iter, number_steps, number_episodes):
    arraySarsa_misalloc = np.empty([n_iter, 4])
    arrayQL_misalloc = np.empty([n_iter, 4])
    arraySarsa_time = np.empty([n_iter, 4])
    arrayQL_time = np.empty([n_iter, 4])

    for i in range(0, n_iter):
        for j, x in enumerate(conditions):
            (arraySarsa_misalloc[i, j], arraySarsa_time[i, j]) = test_exp(
                sarsa, *x, number_episodes=number_episodes, number_steps=number_steps
            )
            (arrayQL_misalloc[i, j], arrayQL_time[i, j]) = test_exp(
                ql, *x, number_episodes=number_episodes, number_steps=number_steps
            )

    # Save to file for further analysis
    np.savez(
        os.path.dirname(os.path.realpath(__file__))
        # Warning: this only works on *nix
        + "/exp3/exp3_"
        + str(number_episodes)
        + "episodes_"
        + str(number_steps)
        + "steps_"
        + datetime.now().strftime("%H-%M-%S"),
        arraySarsa_misalloc,
        arrayQL_misalloc,
        arraySarsa_time,
        arrayQL_time,
    )

    return arraySarsa_misalloc, arrayQL_misalloc, arraySarsa_time, arrayQL_time


def test_exp(
    algorithm,
    earlyRewards,
    capacity_penalty,
    number_steps=100,
    number_episodes=80
):
    props, _, _, times, _, _ = test(
        algorithm,
        capacity_hospital=40,
        number_steps=number_steps,
        number_episodes=number_episodes,
        p_arr_prob=[1, 1, 1],
        doctors=[
            Doctor(0, 0.1),
            Doctor(1, 0.4),
            Doctor(2, 0.1),
        ],
        feature=feature_7,
        rand_rewards=0,
        gamma=0.9,
        alpha=None,
        epsilon=0.1,
        plot_type=None,
        title1="",
        title2="",
        earlyRewards=earlyRewards,
        capacity_penalty=capacity_penalty,
    )
    return misalloc(props), np.median(list(map(lambda l: np.median(l) if len(l)>0 else 0, times)))


def errors(arr):
    """
    Distance from median to 20th and 80th percentile, for error bars
    """
    a = np.median(arr, axis=0)
    return a, np.abs(np.quantile(arr, [0.20, 0.80], axis=0) - a)


def plot(arraySarsa_misalloc, arrayQL_misalloc, arraySarsa_time, arrayQL_time):
    Sarsa_misalloc_points = errors(arraySarsa_misalloc)
    QL_misalloc_points = errors(arrayQL_misalloc)
    Sarsa_time_points = errors(arraySarsa_time)
    QL_time_points = errors(arrayQL_time)

    fig, ax = plt.subplots()

    ind = np.arange(4)  # the x locations for the groups
    width = 0.3  # the width of the bars

	# Misallocations
    p1 = ax.bar(ind, Sarsa_misalloc_points[0], width, yerr=Sarsa_misalloc_points[1], label="SARSA")
    p2 = ax.bar(ind + width, QL_misalloc_points[0], width, yerr=QL_misalloc_points[1], label="Q-learning")

    ax.set_ylabel("Proportion of misallocated patients")
    # ax.set_title("Frequency rate of misallocation")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(("TT", "TF", "FT", "FF"))
    ax.legend()

    autolabel(p1, ax, "left")
    autolabel(p2, ax, "right")

    fig_time, ax_time = plt.subplots()

    ind = np.arange(4)  # the x locations for the groups
    width = 0.3  # the width of the bars

    # Misallocations
    p1 = ax_time.bar(ind, Sarsa_time_points[0], width, yerr=Sarsa_time_points[1], label="SARSA")
    p2 = ax_time.bar(ind + width, QL_time_points[0], width, yerr=QL_time_points[1], label="Q-learning")

    ax_time.set_ylabel("Median waiting time")
    # ax_time.set_title("Frequency rate of misallocation")
    ax_time.set_xticks(ind + width / 2)
    ax_time.set_xticklabels(("TT", "TF", "FT", "FF"))
    ax_time.legend()

    autolabel(p1, ax_time, "left")
    autolabel(p2, ax_time, "right")

    plt.show()


def autolabel(rects, ax, xpos="center"):

    ha = {"center": "center", "right": "left", "left": "right"}
    offset = {"center": 0, "right": 1, "left": -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height, ".0f"),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(offset[xpos] * 3, 3),  # use 3 points offset
            textcoords="offset points",  # in both directions
            ha=ha[xpos],
            va="bottom",
        )



if __name__ == "__main__":
    main(number_tries=2, number_episodes=80, number_steps=100)
