from hospital import *
from learning import *

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


""" Bar plot code (including `autolabel` function) obtained from:https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
"""


feature = feature_12
algorithm = sarsa
number_tries=1
number_episodes=5
number_steps=1


def main(number_tries, number_episodes, number_steps):
    print("Misallocation experiment\n")
    print("Featurisation:", feature)
    print("Algorithm:", algorithm)
    print("Number of episodes per simulation: ", number_episodes)
    print("Number of timesteps per episode: ", number_steps)
    print("Number of trials per set of parameters: ", number_tries)
    # print("Estimated duration:",number_tries*number_steps*number_episodes*0.9 ,"seconds\n")



    if len(sys.argv) == 1:
        data = generate(number_tries, number_steps, number_episodes)
    else:  # Assume only 1 extra argument: npz file containing array
        dict = np.load(str(sys.argv[1]))
        data = tuple(map(lambda x: x[1], dict.items()))
        dict.close()

    # plot(data)


def generate(n_iter, number_steps, number_episodes):
    misalloc_rate = np.empty([n_iter])

    for i in range(0, n_iter):
        misalloc_rate[i] = test_exp(
            algorithm,
            earlyRewards=False,
            capacity_penalty=False,
            number_episodes=number_episodes,
            number_steps=number_steps,
        )

    # Save to file for further analysis (deactivated)
    # np.savez(
    #     os.path.dirname(os.path.realpath(__file__))
    #     + "/exp3/exp3_misalloc"
    #     + str(number_episodes)
    #     + "episodes_"
    #     + str(number_steps)
    #     + "steps_"
    #     + datetime.now().strftime("%H-%M-%S"),
    #     misalloc_rate,
    # )

    return misalloc_rate


def test_exp(
    algorithm, earlyRewards, capacity_penalty, number_steps=100, number_episodes=100
):
    props, *_ = test(
        algorithm,
        capacity_hospital=100,
        number_steps=number_steps,
        number_episodes=number_episodes,
        p_arr_prob=[1, 1, 1],
        doctors=[Doctor(0, 0.1), Doctor(1, 0.1), Doctor(2, 0.1),],
        feature=feature,
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
    return misalloc(props)


def plot(misalloc_array):
    plt.figure()
    plt.hist(misalloc_array)
    plt.show()


if __name__ == "__main__":
    main(number_tries=number_tries, number_episodes=number_episodes, number_steps=number_steps)
