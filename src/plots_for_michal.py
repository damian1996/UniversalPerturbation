import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2 as cv

import consts
import logs_reader
import utils
from consts import *


def plot_mean_results_with_bounds(results, min_results, max_results, percents, path):
    sns.set_style('whitegrid')
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']

    fig, ax = plt.subplots()
    percents2 = [str(p).replace("_", ".") for p in percents]
    modes = ["random", "trained"]
    plt.title(f"Average normalized for {results[0][0]}", color='black')
    for cnt, (algo, algo_results) in enumerate(results):
        vs = [algo_results[percent] for percent in percents]
        min_vs = [min_results[cnt][1][percent] for percent in percents]
        max_vs = [max_results[cnt][1][percent] for percent in percents]
        ax.plot(percents2, vs, colors[cnt % len(colors)], label=modes[cnt])
        ax.fill_between(percents2, min_vs, max_vs, color=colors[cnt % len(colors)], alpha=0.2)

    ax.legend(loc='center left', bbox_to_anchor=(0.75,0.75))
    ax.set_xlabel("Perturbation Magnitude")
    ax.set_ylabel("Normalized Expected Return")

    ax.set_ylim(0, 1.05)
    fig.savefig(path)
    fig.clear()

def plot_mean_results_with_bounds_other_algo(results, min_results, max_results, percents, path, algo, other_algo):
    sns.set_style('whitegrid')
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']

    fig, ax = plt.subplots()
    percents2 = [str(p).replace("_", ".") for p in percents]
    other_algo = "Rainbow" if other_algo.lower() == "rainbow" else other_algo.upper()
    algo = "Rainbow" if algo.lower() == "rainbow" else algo.upper()

    algos = ["Random perturbation", f"Trained perturbation on {algo}", f"Trained perturbation on {other_algo}"]
    plt.title(f"Average normalized for {algo.upper()}", color='black')
    for cnt, (algo, algo_results) in enumerate(results):
        vs = [algo_results[percent] for percent in percents]
        min_vs = [min_results[cnt][1][percent] for percent in percents]
        max_vs = [max_results[cnt][1][percent] for percent in percents]
        ax.plot(percents2, vs, colors[cnt % len(colors)], label=algos[cnt])
        ax.fill_between(percents2, min_vs, max_vs, color=colors[cnt % len(colors)], alpha=0.2)

    if algo == "es":
        ax.legend(loc='center left', bbox_to_anchor=(0.45,0.3))
    else:
        ax.legend(loc='center left', bbox_to_anchor=(0.45,0.88))
    ax.set_xlabel("Perturbation Magnitude")
    ax.set_ylabel("Normalized Expected Return")

    ax.set_ylim(0, 1.05)
    fig.savefig(path)
    fig.clear()

from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_color_table(data1, path, games):
    plt.figure(figsize=(10, 10))   
    ax = plt.gca()
    im = ax.imshow(data1, cmap='bwr')

    x, y = np.arange(len(games)), np.arange(len(games))
    plt.rcParams["axes.grid"] = False

    ax.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(games, fontsize=8, rotation=50)  
    ax.set_yticks(y)
    ax.set_yticklabels(games, fontsize=8, rotation=50)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.20)

    plt.colorbar(im, cax=cax)

    plt.savefig(path)
    plt.close()

def create_transfer_table(results=None, path=None, games=None):
    if results is None:
        colormap = np.arange(53*53).reshape(53, 53)
        colormap = colormap / 2809
    else:
        colormap = results

    plot_color_table(colormap, path, games)

def prepare_results_for_plots(results, magnitudes, games):
    mean_results = { magnitude: 0.0 for magnitude in magnitudes }
    std_results = { magnitude: 0.0 for magnitude in magnitudes }

    for magnitude in magnitudes:
        sumi_mean, sumi_std, cnt = 0.0, 0.0, 0
        for game in games:
            game_results = np.array(results[magnitude][game])
            sumi_mean += np.mean(game_results)
            sumi_std += np.std(game_results)
            cnt += 1
            
        mean_results[magnitude] = (sumi_mean / cnt)
        std_results[magnitude] = (sumi_std / cnt)

    return mean_results, std_results

def draw_mean_plot_for_one_policy(results, algo, magnitudes, games, path, policy_id):
    all_results, left_std_results, right_std_results = [], [], []
    modes = ["random", "trained"]

    for ii, mode in enumerate(modes):
        results_for_mode = results[mode]

        mean_results, std_results = prepare_results_for_plots(results_for_mode, magnitudes, games)

        all_results.append((algo, mean_results))

        left_bounds_with_stddev = {magnitude: max([mean_results[magnitude] - std_results[magnitude], 0]) for magnitude in magnitudes}
        left_std_results.append((algo, left_bounds_with_stddev))
        
        right_bounds_with_stddev = {magnitude: min([mean_results[magnitude] + std_results[magnitude], 1]) for magnitude in magnitudes}
        right_std_results.append((algo, right_bounds_with_stddev))

    plot_mean_results_with_bounds(all_results, left_std_results, right_std_results, magnitudes, path)

def draw_mean_plot_for_other_algo(results, algo, other_algo, magnitudes, games, path, policy_id):
    all_results, left_std_results, right_std_results = [], [], []
        
    modes = ["random", "trained", "other_algo"]
    max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    
    for mode in modes:
        mode_results = results[mode]
        mean_results, std_results = prepare_results_for_plots(mode_results, magnitudes, games)
   
        all_results.append((algo, mean_results))

        left_bounds_with_stddev = {magnitude: max([mean_results[magnitude] - std_results[magnitude], 0]) for magnitude in magnitudes}
        left_std_results.append((algo, left_bounds_with_stddev))
            
        right_bounds_with_stddev = {magnitude: min([mean_results[magnitude] + std_results[magnitude], 1]) for magnitude in magnitudes}
        right_std_results.append((algo, right_bounds_with_stddev))

    plot_mean_results_with_bounds_other_algo(all_results, left_std_results, right_std_results, magnitudes, path, algo, other_algo)

def plot_4_2(results, magnitudes, games, path_for_plots, policy_id, algo):
    path = f'{path_for_plots}/plot_{algo}_policy_{policy_id}.png'
    draw_mean_plot_for_one_policy(results, algo, magnitudes, games, path, policy_id)

def plot_4_3_left(results, magnitudes, games, path_for_plots, policy_id, algo):
    path = f'{path_for_plots}/plot_{algo}_train_policy_{policy_id}_eval_policy_0.png'
    draw_mean_plot_for_one_policy(results, algo, magnitudes, games, path, policy_id)

def plot_4_3_right(results, magnitudes, games, path_for_plots, policy_id, algo, other_algo):
    path = f'{path_for_plots}/plot_{algo}_and_{other_algo}_train_policy_{policy_id}_eval_policy_0.png'
    draw_mean_plot_for_other_algo(results, algo, other_algo, magnitudes, games, path, policy_id)

def draw_transfer_table(results, algo, magnitude, games, policy_id, mode, path):
    transformed_magnitude = str(magnitude).replace(".", "_")
    transfer_table = np.array(results)
            
    path_to_save_transfer_table = f"{path}/transfer_table_{algo}_{transformed_magnitude}_{mode}_policy_{policy_id}.png"
    create_transfer_table(results=transfer_table, path=path_to_save_transfer_table, games=games)

if __name__ == "__main__":
    magnitudes = [0.01, 0.05, 0.1]
    games = ["Pong"] # utils.get_sampled_games()
    path_for_plots = "./plots"
    policy_id = 0
    algo = "rainbow"
    
    results = {
        "trained": {
            0.01: {
                "Pong": [
                    0.15, 0.20, 0.34
                ]
            },
            0.05: {
                "Pong": [
                    0.10, 0.13, 0.17
                ]
            },
            0.1: {
                "Pong": [
                    0.05, 0.05, 0.05
                ]
            }
        },
        "random": {
            0.01: {
                "Pong": [
                    0.70, 0.74, 0.96
                ]
            },
            0.05: {
                "Pong": [
                    0.35, 0.20, 0.37
                ]
            },
            0.1: {
                "Pong": [
                    0.2, 0.2, 0.2
                ]
            }
        },
    }

    # plot 4.2 - ewaluacja trenowanej perturbacji na policy_id = 0 (pierwsza polityka od ubera), trening perturbacji też na polityce 0
    plot_4_2(results, magnitudes, games, path_for_plots, policy_id, algo)

    # plot 4.3 lewy - ewaluacja trenowanej perturbacji na policy_id = 0, ale perturbacja trenowana była na polityce 1
    policy_id = 1
    plot_4_3_left(results, magnitudes, games, path_for_plots, policy_id, algo)

    # plot 4.3 prawy - ewaluacja na polityce 0 z algorytmem A, a na wykresie porownujemy losową perturbację, perturbację wytrenowaną z polityką 1 i z algorytmem A oraz perturbację trenowaną na polityce 1 z algorytmem B
    results = {
        "trained": {
            0.01: {
                "Pong": [
                    0.15, 0.20, 0.34
                ]
            },
            0.05: {
                "Pong": [
                    0.10, 0.13, 0.17
                ]
            },
            0.1: {
                "Pong": [
                    0.05, 0.05, 0.05
                ]
            }
        },
        "random": {
            0.01: {
                "Pong": [
                    0.70, 0.74, 0.96
                ]
            },
            0.05: {
                "Pong": [
                    0.35, 0.20, 0.37
                ]
            },
            0.1: {
                "Pong": [
                    0.2, 0.2, 0.2
                ]
            }
        },
        "other_algo": {
            0.01: {
                "Pong": [
                    0.22, 0.28, 0.35
                ]
            },
            0.05: {
                "Pong": [
                    0.30, 0.20, 0.20
                ]
            },
            0.1: {
                "Pong": [
                    0.25, 0.25, 0.35
                ]
            }
        }
    }
    other_algo = "dqn"
    policy_id = 1
    plot_4_3_right(results, magnitudes, games, path_for_plots, policy_id, algo, other_algo)

    games = ["Pong", "Seaquest", "Breakout"]
    magnitude = 0.01
    mode = "trained"
    transfer_results = np.random.uniform(0, 1, len(games) ** 2).reshape(len(games), len(games))
    draw_transfer_table(transfer_results, algo, magnitude, games, policy_id, mode, path_for_plots)