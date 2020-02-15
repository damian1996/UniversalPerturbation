import numpy as np
import os
import matplotlib.pyplot as plt

import utils
#import results_visualisation.plots as plots
from produce_baselines import *
from consts import *

def measure_power():
    percents = [0.005, 0.008, 0.01]

    r_paths = ["colormap_fix_random_0_005.npy", "colormap_fix_random_0_008.npy", "colormap_fix_random_0_01.npy"]
    t_paths = ["colormap_fix_trained_0_005.npy", "colormap_fix_trained_0_008.npy", "colormap_fix_trained_0_01.npy"]

    for ii in range(len(percents)):
        print(f"Case {percents[ii]}")
        a = np.load(os.path.join("colormaps", r_paths[ii]))
        b = np.load(os.path.join("colormaps", t_paths[ii]))

        print(f"Z przekatna dla {percents[ii]} {np.mean(a)} {np.mean(b)}")

        nr_w_cross = (a.shape[0] ** 2) - a.shape[0]
        sa, sb = 0., 0.
        
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                if i == j:
                    continue

                sa += a[i,j]
                sb += b[i,j]

        sa = sa / nr_w_cross
        sb = sb / nr_w_cross

        print(f"Bez przekatna {percents[ii]} {sa} {sb}")


def draw_pie_chart(names, round_chart, mode, s_max_noise):
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.linspace(0., 1., len(round_chart)))

    round_chart = np.array(round_chart)
    patches, texts = plt.pie(round_chart, colors=colors, startangle=140, radius=1.2)
    
    porcent = (round_chart/round_chart.sum()) * 100
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(names, porcent)]

    sort_legend = False
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, round_chart),
                                            key=lambda x: x[2],
                                            reverse=True))

    plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.),
                fontsize=8, title=mode)

    plt.savefig(f'pies/plot__{mode}_{s_max_noise}.png', bbox_inches='tight')

    plt.axis('equal')
    plt.close()

def plot_transfer_tables(results_with_pert, noises, envs):
    trained_policy = read_baselines_from_files("trained", envs)
    random_policy = read_baselines_from_files("random", envs)
    print("random_policy", random_policy)
    print("trained_policy", trained_policy)
    print("policies summed ", np.sum(np.array(trained_policy)), np.sum(np.array(random_policy)))

    for j, results_path in enumerate(results_with_pert):
        nr_envs = len(envs)
        colormap = np.zeros((nr_envs, nr_envs))
        
        path_parts = results_path.split("_")
        mode = path_parts[2]
        s_max_noise = noises[j]

        with open(os.path.join("colormaps", results_path), "r") as f:
            lines = f.readlines()

            all_sum = 0
            for i, line in enumerate(lines):
                ii, jj = (i // nr_envs), (i % nr_envs)
                line = line.strip().replace("[", "").replace("]", "")
                parts = line.split(" ")
                r1, r2, r3 = float(parts[2]), float(parts[3]), float(parts[4])
                
                normalized = utils.normalize((r1, r2, r3), random_policy[jj], trained_policy[jj])
                colormap[ii, jj] = normalized

        print(mode, s_max_noise, " ", np.mean(colormap))
        names = []
        round_chart = []
        for iii in range(nr_envs):
            names.append(envs[iii])
            round_chart.append(np.mean(colormap[iii, :]))
            # print("Game", envs[iii], np.mean(colormap[:, iii]))
        
        draw_pie_chart(names, round_chart, mode, s_max_noise)

        np.save(open(f'colormaps/colormap_fix_{mode}_{s_max_noise}.npy', 'wb'), colormap)
        #plots.normalized_color_table(colormap, os.path.join("colormaps", f"colormap_fix_{mode}_{s_max_noise}.png"))

'''
results_with_pert = [
    "results_colormap_random_0_005_best_policy_16_84_84_4.txt",
    "results_colormap_trained_0_005_best_policy_16_84_84_4.txt"
]
noises = ["0_005", "0_005"]
plot_transfer_tables(results_with_pert, noises)


results_with_pert = [
    "results_colormap_random_0_008_best_policy_16_84_84_4.txt",
    "results_colormap_trained_0_008_best_policy_16_84_84_4.txt"
]
noises = ["0_008", "0_008"]
plot_transfer_tables(results_with_pert, noises)



results_with_pert = [
    "results_colormap_random_0_005_best_policy_16.txt",
    "results_colormap_trained_0_005_best_policy_16.txt"
]
noises = ["0_005", "0_005"]
plot_transfer_tables(results_with_pert, noises)


results_with_pert = [
    "results_colormap_random_0_008_best_policy_16.txt",
    "results_colormap_trained_0_008_best_policy_16.txt"
]
noises = ["0_008", "0_008"]
plot_transfer_tables(results_with_pert, noises)



results_with_pert = [
    "results_colormap_random_0_008_best_policy_lr_0_005.txt",
    "results_colormap_trained_0_008_best_policy_lr_0_005.txt"
]
noises = ["0_008", "0_008"]
plot_transfer_tables(results_with_pert, noises)


results_with_pert = [
    "results_colormap_random_0_008_best_policy_lr_0_1.txt",
    "results_colormap_trained_0_008_best_policy_lr_0_1.txt"
]
noises = ["0_008", "0_008"]
plot_transfer_tables(results_with_pert, noises)
'''