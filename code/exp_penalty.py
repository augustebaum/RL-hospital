from hospital import *
from learning import *
from experiments_extra_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

""" Bar plot code (including `autolabel` function) obtained from:https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
"""
def main(n_iter = 1):
    conditions = [(True, True), (True, False), (False,True), (False, False)]
    arraySarsa = np.empty([n_iter, 4])
    arrayQL = np.empty([n_iter, 4])

    for i in range(0, n_iter):
        for j, x in enumerate(conditions):
            arraySarsa[i, j] = test_exp(sarsa, *x)
            arrayQL[i, j]    = test_exp(ql, *x)

    SarsaMeans = np.mean(arraySarsa, axis=0)
    SarsaStds = np.std(arraySarsa, axis=0)

    QLMeans = np.mean(arrayQL, axis=0)
    QLStds = np.std(arrayQL, axis=0)

    return SarsaMeans, SarsaStds, QLMeans, QLStds

def test_exp(algorithm, earlyRewards, capacity_penalty):
    x, *_ = test(algorithm,
                 capacity_hospital = 100,
                 number_steps = 100,
                 number_episodes = 100,
                 p_arr_prob = [1, 1, 1, 1, 1, 1],
                 doctors = [Doctor(0, 0.1),
                 Doctor(1, 0.1),
                 Doctor(2, 0.9),
                 Doctor(3, 0.1),
                 Doctor(4, 0.5),
                 Doctor(5, 0.1)],
                 feature = feature_7,
                 rand_rewards = 0,
                 gamma = 0.9,
                 alpha = None,
                 epsilon = 0.1,
                 plot_type = None,
                 title1 = "(1.1)Sarsa + earlyReward + Capacity_penalty",
                 title2 = "(1.1)Reward evolution for the picture above",
                 earlyRewards = earlyRewards,
                 capacity_penalty = capacity_penalty)
    return misalloc(x)

def autolabel(rects, xpos='center'):
 
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height, '.0g'),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


if __name__ == '__main__':
    SarsaMeans, SarsaStds, QLMeans, QLStds = main(n_iter = 3)

    fig, ax = plt.subplots()

    ind = np.arange(4)    # the x locations for the groups
    width = 0.3         # the width of the bars
    p1 = ax.bar(ind, SarsaMeans, width, yerr=SarsaStds, label='SARSA')
    p2 = ax.bar(ind + width, QLMeans, width, yerr=QLStds, label='Q-Learning')

    ax.set_ylabel('Frequency Rate')
    ax.set_title('Frequency rate of misallocation ')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('TT', 'TF', 'FT', 'FF'))
    ax.legend()


    autolabel(p1, "left")
    autolabel(p2, "right")

    plt.show()

