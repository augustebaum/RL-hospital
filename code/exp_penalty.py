from experiments_extra_functions import *

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
                 earlyRewards = earlyRewards, capacity_penalty = capacity_penalty)
    return misalloc(x)


if __name__ == '__main__':
    main(n_iter = 5)
