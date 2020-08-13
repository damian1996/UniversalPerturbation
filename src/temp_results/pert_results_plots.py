import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import math
from numpy import *
import numpy as np
import seaborn as sns; sns.set()
import cv2
import os
import sys
from operator import attrgetter
import sys, inspect
import neptune_client
import utils
import random_perturbation_generator as rpg
from consts import *
import produce_baselines as pb


# def draw_perturbation(filenames, inv):
#     for i, filename in enumerate(filenames):
#         pert = np.load(f"results_visualisation/results_to_draw/{filename}")

#         cv2.imwrite(f"results_visualisation/results_to_draw/pert{i}_{inv[i]}_czb.png", pert[:, :, :1]*256*(1./inv[i]))


def plots_three_types_results(random_pert, trained_pert, percents, algos):
    sns.set_style('whitegrid')
    
    results = [random_pert, trained_pert]
    types = ["random", "trained"]
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    plt.title(f"random and trained plots", color='black')

    cnt = -1
    for algo_id, algo in enumerate(algos):
        for type_id, res_type in enumerate(types):
            k = percents
            v = results[type_id][algo_id]
            label = f"{res_type} {algo}"
            cnt += 1
            print(k, v)
            plt.plot(k, v, colors[cnt % len(colors)], label=label)

    plt.legend(loc='center left', bbox_to_anchor=(0.75,0.6))
    
    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Model Results")

    plt.xticks(percents)
    x_values = np.linspace(0.0, np.max(np.array(percents)), len(percents) + 1)[1:]
    print(x_values)
    plt.xticks(x_values, percents)#, rotation='vertical')
    plt.ylim(0, 1.05)
    plt.savefig(f'final_plots/plot_test.png')
    plt.close()

def sns_test():
    sns.set_style('whitegrid')
    fmri = sns.load_dataset("fmri")
    ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
    # df = sns.load_dataset('iris')
    # sns_plot = sns.pairplot(df, hue='species', size=2.5)
    ax.figure.savefig("final_plots/output.png")

def make_variance_plots(algo, game, policy_id):
    uppercase_game = f"{game[0].upper()}{game[1:]}"
    print(uppercase_game)
    trained_path = f"./final_results/trained/{algo}/results/{uppercase_game}"
    random_path = f"./final_results/random/{algo}/results/{uppercase_game}"

    colors = ['purple', 'blue', 'brown', 'green', 'red', 'orange', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    paths = [trained_path, random_path]
    modes = ["trained", "random"]
    
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    
    cnt = 0
    org_path = f"all_baselines_from_uber/trained/{uppercase_game}_{algo}_policy_score.npy"
    original_result = [np.load(org_path)[policy_id]]

    for ii, path in enumerate(paths):
        all_percents, all_results, all_other_results = [], [], []
        perts = sorted(os.listdir(path))
        print(perts)
        results, other_results = {}
        for nr_pert in perts:
            pert_path = f"{path}/{nr_pert}"
            percents = os.listdir("pert_path")

            for percent in percents:
                if ".py" in percent:
                    continue
                
                if nr_pert == 0:
                    results[percent] = []
                    other_results[percent] = []

                percent_path = f"{pert_path}/{percent}"
                games = os.listdir(percent_path)
                
                all_percents.append(percent)
                results[percent].append([np.load(f"{percent_path}/{game}")[policy_id] for game in games if game == uppercase_game])
                other_results[percent].append([np.load(f"{percent_path}/{game}")[policy_id] for game in games if game != uppercase_game])
            
            if cnt == 0:
                original_results = original_result * len(all_results)
                x = all_percents

        inf = 1000000.0
        game_lower, game_upper, others_lower, others_upper = [inf] * len(x) , [-inf] * len(x), [inf] * len(x), [-inf] * len(x)
        res, others_res = [0.0] * len(x), [0.0] * len(x)

        for nr_percent, percent in enumerate(all_percents):
            for results_for_pert in results[percent]:
                res[nr_percent] += results_for_pert[0]
                
                game_lower[nr_percent] = min(game_lower[nr_percent], results_for_pert[0])
                game_upper[nr_percent] = max(game_upper[nr_percent], results_for_pert[0])
            
            for other_results_for_pert in other_results[percent]:
                others_res[nr_percent] += np.mean(np.array(other_results_for_pert))
                
                others_lower[nr_percent] = min(others_lower[nr_percent], np.mean(np.array(other_results_for_pert)))
                others_upper[nr_percent] = max(others_upper[nr_percent], np.mean(np.array(other_results_for_pert)))
            

        cnt += 1
        ax.plot(x, res, colors[cnt % len(colors)], label=f"{modes[ii]} {game.lower()}")
        ax.fill_between(x, game_lower, game_upper, alpha=0.2)

        cnt += 1
        ax.plot(x, others_res, colors[cnt % len(colors)], label=f"{modes[ii]} other games")
        ax.fill_between(x, others_lower, others_upper, alpha=0.2)

    ax.plot(x, original, colors[cnt % len(colors)], label=f"original")

    x_values = np.linspace(0.0, np.max(np.array(x)), len(x) + 1)[1:]
    plt.xticks(x_values, x)#, rotation='vertical')

    plt.legend(loc='center left', bbox_to_anchor=(0.65,0.6))
    plt.savefig('test.png')

if __name__ == "__main__":
    sns_test()