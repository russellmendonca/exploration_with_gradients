import argparse

import numpy as np

from method import (
    Mlp,
    collect_data_and_assign_to_ensembles,
    compute_traj_fast,
    fit_dynamics,
    load_array_and_plot,
)
from wrapper_envs import HopperEnv
import torch
import matplotlib.pyplot as plt


def update_counts_from_new_transitions(
    initial_transitions,
    num_ensembles,
    Xs,
    bins_angular_1,
    bins_angular_2,
    bins_angular_3,
    bins_angular_4,
    angular_logs_1,
    angular_logs_2,
    angular_logs_3,
    angular_logs_4,
):
    # handles aggregation across time
    for i in range(num_ensembles):
        X, U, X_prime = initial_transitions[i]
        Xs.append(X)
        X = np.concatenate(Xs)
    Angles_1 = X[:, 0]
    Angles_2 = X[:, 1]
    Angles_3 = X[:, 2]
    Angles_4 = X[:, 3]

    counts_angular_1 = plt.hist(Angles_1, bins_angular_1)[0]
    counts_angular_2 = plt.hist(Angles_2, bins_angular_2)[0]
    counts_angular_3 = plt.hist(Angles_3, bins_angular_3)[0]
    counts_angular_4 = plt.hist(Angles_4, bins_angular_4)[0]

    angular_logs_1.append(counts_angular_1)
    angular_logs_2.append(counts_angular_2)
    angular_logs_3.append(counts_angular_3)
    angular_logs_4.append(counts_angular_4)


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_plots", type=bool, default=False)
    args = parser.parse_args()

    bins_angular_1 = np.arange(-1.3, 1.3, 0.01)
    bins_angular_2 = np.arange(0, 1.5, 0.01)
    bins_angular_3 = np.arange(-5.3, 1.100, 0.01)
    bins_angular_4 = np.arange(-2.8, 0.1, 0.01)
    T = 100
    latent_dim = 12
    num_ensembles = 5
    num_seeds = 5

    env = HopperEnv()

    Us = [
        np.concatenate([env.action_space.sample().reshape(1, -1) for i in range(T)], 0)
        for _ in range(int(10))
    ]
    initial_transitions = collect_data_and_assign_to_ensembles(env, num_ensembles, Us)

    num_angular_bins_1 = len(bins_angular_1)
    num_angular_bins_2 = len(bins_angular_2)
    num_angular_bins_3 = len(bins_angular_3)
    num_angular_bins_4 = len(bins_angular_4)

    if args.make_plots:
        load_array_and_plot(
            "hopper/angular_logs_random_1.npy",
            "hopper/angular_logs_ours_1.npy",
            "hopper/angular_1.png",
            num_angular_bins_1,
            "Angular 1 Component of State",
        )
        load_array_and_plot(
            "hopper/angular_logs_random_2.npy",
            "hopper/angular_logs_ours_2.npy",
            "hopper/angular_2.png",
            num_angular_bins_2,
            "Angular 2 Component of State",
        )
        load_array_and_plot(
            "hopper/angular_logs_random_3.npy",
            "hopper/angular_logs_ours_3.npy",
            "hopper/angular_3.png",
            num_angular_bins_3,
            "Angular 3 Component of State",
        )
        load_array_and_plot(
            "hopper/angular_logs_random_4.npy",
            "hopper/angular_logs_ours_4.npy",
            "hopper/angular_4.png",
            num_angular_bins_4,
            "Angular 4 Component of State",
        )
    else:
        all_angular_logs_random_1 = []
        all_angular_logs_random_2 = []
        all_angular_logs_random_3 = []
        all_angular_logs_random_4 = []

        for i in range(num_seeds):
            np.random.seed(i)
            angular_logs_random_1 = []
            angular_logs_random_2 = []
            angular_logs_random_3 = []
            angular_logs_random_4 = []

            Xs = []
            update_counts_from_new_transitions(
                initial_transitions,
                num_ensembles,
                Xs,
                bins_angular_1,
                bins_angular_2,
                bins_angular_3,
                bins_angular_4,
                angular_logs_random_1,
                angular_logs_random_2,
                angular_logs_random_3,
                angular_logs_random_4,
            )
            for epoch in range(500):
                Us = [
                    np.concatenate(
                        [env.action_space.sample().reshape(1, -1) for i in range(T)], 0
                    )
                    for i in range(1)
                ]
                transitions_new = collect_data_and_assign_to_ensembles(
                    env, num_ensembles, Us
                )
                update_counts_from_new_transitions(
                    transitions_new,
                    num_ensembles,
                    Xs,
                    bins_angular_1,
                    bins_angular_2,
                    bins_angular_3,
                    bins_angular_4,
                    angular_logs_random_1,
                    angular_logs_random_2,
                    angular_logs_random_3,
                    angular_logs_random_4,
            )
            all_angular_logs_random_1.append(np.array(angular_logs_random_1))
            all_angular_logs_random_2.append(np.array(angular_logs_random_2))
            all_angular_logs_random_3.append(np.array(angular_logs_random_3))
            all_angular_logs_random_4.append(np.array(angular_logs_random_4))
        all_angular_logs_random_1 = np.array(all_angular_logs_random_1)
        all_angular_logs_random_2 = np.array(all_angular_logs_random_2)
        all_angular_logs_random_3 = np.array(all_angular_logs_random_3)
        all_angular_logs_random_4 = np.array(all_angular_logs_random_4)

        np.save("hopper/angular_logs_random_1.npy", all_angular_logs_random_1)
        np.save("hopper/angular_logs_random_2.npy", all_angular_logs_random_2)
        np.save("hopper/angular_logs_random_3.npy", all_angular_logs_random_3)
        np.save("hopper/angular_logs_random_4.npy", all_angular_logs_random_4)

        import copy

        all_angular_logs_ours_1 = []
        all_angular_logs_ours_2 = []
        all_angular_logs_ours_3 = []
        all_angular_logs_ours_4 = []
        for _ in range(num_seeds):
            angular_logs_ours_1 = []
            angular_logs_ours_2 = []
            angular_logs_ours_3 = []
            angular_logs_ours_4 = []
            Xs = []
            transitions = copy.deepcopy(initial_transitions)

            update_counts_from_new_transitions(
                initial_transitions,
                num_ensembles,
                Xs,
                bins_angular_1,
                bins_angular_2,
                bins_angular_3,
                bins_angular_4,
                angular_logs_ours_1,
                angular_logs_ours_2,
                angular_logs_ours_3,
                angular_logs_ours_4,
            )

            # initial training
            phis = [
                Mlp(
                    hidden_sizes=[32],
                    output_size=latent_dim,
                    input_size=env.observation_space.low.size,
                )
                for _ in range(num_ensembles)
            ]
            Bs_mlp = [
                Mlp(
                    hidden_sizes=[],
                    output_size=latent_dim,
                    input_size=env.action_space.low.size,
                )
                for _ in range(num_ensembles)
            ]
            optimizers = [
                torch.optim.Adam(
                    list(phis[i].parameters()) + list(Bs_mlp[i].parameters())
                )
                for i in range(num_ensembles)
            ]
            fit_dynamics(
                initial_transitions,
                phis,
                Bs_mlp,
                optimizers,
                iterations=100,
                num_ensembles=num_ensembles,
            )
            As = [phis[i].last_fc.weight.detach() for i in range(num_ensembles)]
            Bs = [Bs_mlp[i].last_fc.weight.detach() for i in range(num_ensembles)]

            for epoch in range(500):
                Us = []
                # compute optimized disagreement trajectories
                indices = np.random.choice(num_ensembles, num_ensembles, replace=False)
                phis_shuffled = [phis[i] for i in indices]
                Bs_shuffled = [Bs[i] for i in indices]
                As_shuffled = [As[i] for i in indices]
                u = compute_traj_fast(
                    As_shuffled,
                    Bs_shuffled,
                    T,
                    phis_shuffled,
                    tol=1,
                    max_iterations=30,
                    x_rescaling=1.15,
                    u_rescaling=3,
                )
                Us.append(u)
                transitions_new = collect_data_and_assign_to_ensembles(
                    env, num_ensembles, Us
                )

                # update current X
                update_counts_from_new_transitions(
                    transitions_new,
                    num_ensembles,
                    Xs,
                    bins_angular_1,
                    bins_angular_2,
                    bins_angular_3,
                    bins_angular_4,
                    angular_logs_ours_1,
                    angular_logs_ours_2,
                    angular_logs_ours_3,
                    angular_logs_ours_4,
                )

                # update transitions with newly collected transitions
                for i in range(num_ensembles):
                    X, U, X_prime = transitions_new[i]
                    X_, U_, X_prime_ = transitions[i]
                    transitions[i] = [
                        np.concatenate((X_, X)),
                        np.concatenate((U_, U)),
                        np.concatenate((X_prime_, X_prime)),
                    ]

                fit_dynamics(
                    transitions,
                    phis,
                    Bs_mlp,
                    optimizers,
                    iterations=20,
                    num_ensembles=num_ensembles,
                )
                As = [phis[i].last_fc.weight.detach() for i in range(num_ensembles)]
                Bs = [Bs_mlp[i].last_fc.weight.detach() for i in range(num_ensembles)]
            all_angular_logs_ours_1.append(np.array(angular_logs_ours_1))
            all_angular_logs_ours_2.append(np.array(angular_logs_ours_2))
            all_angular_logs_ours_3.append(np.array(angular_logs_ours_3))
            all_angular_logs_ours_4.append(np.array(angular_logs_ours_4))
        all_angular_logs_ours_1 = np.array(all_angular_logs_ours_1)
        all_angular_logs_ours_2 = np.array(all_angular_logs_ours_2)
        all_angular_logs_ours_3 = np.array(all_angular_logs_ours_3)
        all_angular_logs_ours_4 = np.array(all_angular_logs_ours_4)

        np.save("hopper/angular_logs_ours_1.npy", all_angular_logs_ours_1)
        np.save("hopper/angular_logs_ours_2.npy", all_angular_logs_ours_2)
        np.save("hopper/angular_logs_ours_3.npy", all_angular_logs_ours_3)
        np.save("hopper/angular_logs_ours_4.npy", all_angular_logs_ours_4)

