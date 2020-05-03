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
    # for arr in data:
        # arr = arr[~np.isnan(arr).any(axis=1)]

    data = map(lambda arr: arr[~np.isnan(arr).any(axis=1)], data)

    plot(*data)


def generate(n_iter, number_steps, number_episodes):
    arraySarsa = np.empty([n_iter, 4])
    arrayQL = np.empty([n_iter, 4])

    for i in range(0, n_iter):
        for j, x in enumerate(conditions):
            arraySarsa[i, j] = test_exp(
                sarsa, *x, number_episodes=number_episodes, number_steps=number_steps
            )
            arrayQL[i, j] = test_exp(
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
        arraySarsa,
        arrayQL,
    )

    return arraySarsa, arrayQL
    # return SarsaMeans, SarsaStds, QLMeans, QLStds


def test_exp(
    algorithm,
    earlyRewards,
    capacity_penalty,
    number_steps=100,
    number_episodes=100
):
    _, _, _, times, _ = test(
        algorithm,
        capacity_hospital=100,
        number_steps=number_steps,
        number_episodes=number_episodes,
        p_arr_prob=[1, 1, 1],
        doctors=[
            Doctor(0, 0.1),
            Doctor(1, 0.1),
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
    return np.median(list(map(lambda l: np.median(l) if len(l)>0 else 0, times)))


def errors(arr):
    """
    Distance from median to 20th and 80th percentile, for error bars
    """
    a = np.median(arr, axis=0)
    return a, np.abs(np.quantile(arr, [0.20, 0.80], axis=0) - a)


def plot(arraySarsa, arrayQL):
    # SarsaMeans = np.mean(arraySarsa, axis=0)
    # SarsaStds = np.std(arraySarsa, axis=0)

    # QLMeans = np.mean(arrayQL, axis=0)
    # QLStds = np.std(arrayQL, axis=0)

    Sarsa_points = errors(arraySarsa)
    QL_points = errors(arrayQL)

    fig, ax = plt.subplots()

    ind = np.arange(4)  # the x locations for the groups
    width = 0.3  # the width of the bars

    p1 = ax.bar(ind, Sarsa_points[0], width, yerr=Sarsa_points[1], label="SARSA")
    p2 = ax.bar(ind + width, QL_points[0], width, yerr=QL_points[1], label="Q-learning")
    # p2 = ax.bar(ind + width, QLMeans, width, yerr=QLStds, label="Q-Learning")

    ax.set_ylabel("Median waiting time")
    # ax.set_title("Frequency rate of misallocation")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(("TT", "TF", "FT", "FF"))
    ax.legend()

    autolabel(p1, ax, "left")
    autolabel(p2, ax, "right")

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
    main(number_tries=1, number_episodes=50, number_steps=100)
