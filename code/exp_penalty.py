import experiment_extrafunctions

def test_exp(algorithm, earlyRewards, capacity_penalty):
    x, *_ = test(algorithm,
        capacity_hospital,
        number_steps,
        number_episodes,
        p_arr_prob,
        doctors_1,
        feature,
        total_reward_per_episode_r,
        gamma = 0.9,
        alpha = None,
        epsilon = 0.1,
        plot_type = None,
        title1 = "(1.1)Sarsa + earlyReward + Capacity_penalty",
        title2 = "(1.1)Reward evolution for the picture above",
        earlyRewards = earlyRewards, capacity_penalty = capacity_penalty)

    return misalloc(x)



conditions = [(True, True), (True, False), (False,True), (False, False)]

n = 3

arraySarsa = np.empty([n, 4])
for i in range(0, n):
    for j, x in enumerate(conditions):
        array[i, j] = test_exp(sarsa, x)

arrayQL = np.empty([n, 4])
for i in range(0, n):
    for j, x in enumerate(conditions):
        array[i, j] = test_exp(ql, x)

SarsaMeans = np.mean(arraySarsa, axis=0)
SarsaStds = np.std(arraySarsa, axis=0)

QLMeans = np.mean(arrayQL, axis=0)
QLStds = np.std(arrayQL, axis=0)
