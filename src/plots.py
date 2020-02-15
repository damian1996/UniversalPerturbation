import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import math
from numpy import *
import numpy as np
import seaborn as sns
import cv2
import os
import sys
from operator import attrgetter
import sys, inspect
import neptune_client
import perturbation_test
import utils
import random_perturbation_generator as rpg
from consts import *
import produce_baselines as pb


def draw_perturbation(filenames, inv):
    for i, filename in enumerate(filenames):
        pert = np.load(f"results_visualisation/results_to_draw/{filename}")

        cv2.imwrite(f"results_visualisation/results_to_draw/pert{i}_{inv[i]}_czb.png", pert[:, :, :1]*256*(1./inv[i]))


def plot_results_for_one_game(res1, res2, key, run_id):
    sns.set_style('whitegrid')
    print("GAME", key)
    
    colors = ['orange', 'blue', 'green', 'red', 'purple']
    plt.title(f"game policy {run_id}", color='black')

    plt.plot(res1["percents"], res1["ys"], colors[0], label=f"84-84-4")
    plt.plot(res2["percents"], res2["ys"], colors[1], label=f"84-84-1")

    plt.legend(loc='center left', bbox_to_anchor=(0.87,0.6))

    # plt.xticks(x)
    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Model Results")
    max_y = max(np.max(np.array(res1["ys"])), np.max(np.array(res2["ys"])))
    max_y += int(.1 * max_y)

    plt.ylim(0, max_y)
    plt.savefig(f'results_visualisation/results_to_draw/plot_{key}_{run_id}.png')
    plt.close()

    print("plot finished")


def multi_learning_rate_plot(plot_data, game, run_id, mode):
    sns.set_style('whitegrid')
    print("GAME", game)
    
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    plt.title(f"game policy {run_id}", color='black')

    maxi = 0
    for i, (k, v) in enumerate(plot_data.items()):
        print("YAHHAHAHAHA", i, len(colors))
        idx = i % len(colors)
        print(colors[idx])
        plt.plot(v["percents"], v["ys"], colors[idx], label=f"{k}")
        maxi = max(np.array(np.max(v["ys"])), maxi)

    plt.legend(loc='center left', bbox_to_anchor=(0.87,0.6))
    
    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Model Results")

    plt.ylim(0, maxi)
    plt.savefig(f'results_visualisation/results_to_draw/plot_{game}_{run_id}_{mode}_lrs.png')
    plt.close()

    print("plot finished")


def preprocessing(results):
    # Game = collections.namedtuple('Game', 'x y run_id noise_percent')
    results2 = sorted(results, key=attrgetter('noise_percent'))

    ys = [res.y for res in results2]
    run_ids = [res.run_id for res in results2]
    noise_percents = [res.noise_percent for res in results2]

    return {"ys": ys, "run_ids": run_ids, "percents" :noise_percents}

def make_plots(lr, games):
    exps1 = neptune_client.get_experiments_games(lr, games, case=False)
    exps2 = neptune_client.get_experiments_games(lr, games, case=True)

    for i, game in enumerate(games):
        results1 = neptune_client.parse_results_from_experiments(exps1.copy(), game)
        results2 = neptune_client.parse_results_from_experiments(exps2.copy(),game)

        prep_0_0 = preprocessing(results1[0])
        prep_0_1 = preprocessing(results1[1])
        prep_0_2 = preprocessing(results1[2])

        prep_1_0 = preprocessing(results2[0])
        prep_1_1 = preprocessing(results2[1])
        prep_1_2 = preprocessing(results2[2])

        plot_results_for_one_game(prep_0_0, prep_1_0, game, 1)
        plot_results_for_one_game(prep_0_1, prep_1_1, game, 2)
        plot_results_for_one_game(prep_0_2, prep_1_2, game, 3)


def get_lrs_in_range(inputs, left, right):
    results = {}
    for k, v in inputs.items():
        if k >= left and k <= right:
            new_element = {"ys": [], "run_ids": [], "percents": []}
            nr_noises = len(v["ys"])

            for nr in range(nr_noises):
                if v["percents"][nr] in ['0.003', '0.006', '0.01']:
                    new_element["ys"].append(v["ys"][nr])
                    new_element["percents"].append(v["percents"][nr])
                    new_element["run_ids"].append(v["run_ids"][nr])

            if len(new_element["ys"]) == len(['0.003', '0.006', '0.01']):
                results[k] = new_element

    print(results)

    return results        

def make_lrs_plots(games):
    results = neptune_client.get_experiments_lrs(games)
    
    plot_data_1, plot_data_2, plot_data_3 = {}, {}, {}
    for game in games:
        parsed_results = neptune_client.parse_results_for_lrs_experiments(results, game)

        for k, v in parsed_results.items():
            one, two, three = preprocessing(v[0]),  preprocessing(v[1]), preprocessing(v[2])
            plot_data_1[k] = one
            plot_data_2[k] = two
            plot_data_3[k] = three

        plot_data_1_smaller = get_lrs_in_range(plot_data_1, 0.0, 0.00099)
        plot_data_1_greather = get_lrs_in_range(plot_data_1, 0.001, 0.1)
        
        plot_data_2_smaller = get_lrs_in_range(plot_data_2, 0.0, 0.00099)
        plot_data_2_greather = get_lrs_in_range(plot_data_2, 0.001, 0.1)
        
        plot_data_3_smaller = get_lrs_in_range(plot_data_3, 0.0, 0.00099)
        plot_data_3_greather = get_lrs_in_range(plot_data_3, 0.001, 0.1)


        multi_learning_rate_plot(plot_data_1_smaller, game, 1, "smaller")
        multi_learning_rate_plot(plot_data_1_greather, game, 1, "greather")

        multi_learning_rate_plot(plot_data_2_smaller, game, 2, "smaller")
        multi_learning_rate_plot(plot_data_2_greather, game, 2, "greather")

        multi_learning_rate_plot(plot_data_3_smaller, game, 3, "smaller")
        multi_learning_rate_plot(plot_data_3_greather, game, 3, "greather")


def plot_sticky_vs_non_sticky(games):
    repeats = 5
    capitalized_games = [f"{game[0].capitalize()}{game[1:]}" for game in games]
    #pb.produce_baselines_from_uber("random", True, games=capitalized_games)
    #pb.produce_baselines_from_uber("trained", False, games=capitalized_games)

    noise_percents = [0.]
    all_noises = [None]
    #noise_percents_temp = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    #all_noises_temp = [rpg.generate_random_perturbation(percent) for percent in noise_percents_temp]

    #noise_percents.extend(noise_percents_temp)
    #all_noises.extend(all_noises_temp)
    '''
    for (percent, noise) in zip(noise_percents, all_noises):
        percent_str = str(percent).replace('.', '_')
        path = f"sticky_results/noise_{percent_str}.npy"
        np.save(path, noise)
    '''

    sticky_actions_probs = [0.0]
    for sticky_action_prob in sticky_actions_probs:
        print("STICKY_PROB", sticky_action_prob)
        all_algos_results = {}
        full_results = []
        print(algos[2:])

        for algo in algos[2:]:
            random_action_scores = pb.read_baselines_from_files("random", capitalized_games, algo)
            trained_action_scores = pb.read_baselines_from_files("trained", capitalized_games, algo)
            print(f"random_action_scores for {algo}: {random_action_scores}")
            print(f"trained_action_scores for {algo}: {trained_action_scores}")

            all_results_for_algo = []
            for game_id, game in enumerate(games):
                print(f"{algo} {game}")
                env = f"{game[0].capitalize()}{game[1:].lower()}NoFrameskip-v4"
                rand_act_score, trained_act_score = random_action_scores[game_id], trained_action_scores[game_id]

                results_per_game = []
                for (percent, noise) in zip(noise_percents, all_noises):
                    print(f"NOISE PERCENT {percent}")
                    result_mean = 0.
                    for nr in range(repeats):
                        results = perturbation_test.run_experiments_for_env(noise, env, algo=algo, sticky_action_prob=sticky_action_prob)
                        results = [results[0][0], results[1][0], results[2][0]]   
                        
                        print("No normalization", results)
                        normalized_result = utils.normalize(results, rand_act_score, trained_act_score)
                        print("After normalization", normalized_result)
                        result_mean += normalized_result
                
                    result_mean /= 5
                    results_per_game.append(result_mean)
                
                print("Results per game", results_per_game)
                all_results_for_algo.append(np.array(results_per_game))
                
                path = f"sticky_results/{game}_{algo}_random_noise_{sticky_action_prob}_test.npy"
                np.save(path, np.array(results_per_game))
            
            print("Results per algo", all_results_for_algo)
            full_results.append(all_results_for_algo)
            results_for_algo_array = np.array([np.array(per_game) for per_game in all_results_for_algo])
            mean_results_for_algo = np.mean(results_for_algo_array, axis=0)
            all_algos_results[algo] = mean_results_for_algo.tolist()
            
            print("Mean results for algo", mean_results_for_algo)

            path = f"sticky_results/all_games_{algo}_random_noise_{sticky_action_prob}_test.npy"
            np.save(path, np.array(mean_results_for_algo))

        print("Final results mwhahahha", all_algos_results.items())
        
        plot_for_algos(all_algos_results, noise_percents, sticky_action_prob)
        
        for game_id, game in enumerate(games):
            game_plot_data = {}
            for i, algo_results in enumerate(full_results):
                game_plot_data[algos[i]] = algo_results[game_id]

            plot_for_algos(game_plot_data, noise_percents, sticky_action_prob, game=game)

        print("Plots done")

def get_results_for_no_perturbation_sticky(games):
    repeats = 5
    capitalized_games = [f"{game[0].capitalize()}{game[1:]}" for game in games]
    #pb.produce_baselines_from_uber("random", True, games=capitalized_games)
    #pb.produce_baselines_from_uber("trained", False, games=capitalized_games)

    noise_percents = [0.]
    all_noises = [None]

    for (percent, noise) in zip(noise_percents, all_noises):
        percent_str = str(percent).replace('.', '_')

    sticky_actions_probs = [0.25, 0.0]
    for sticky_action_prob in sticky_actions_probs:
        print("STICKY_PROB", sticky_action_prob)
        all_algos_results = {}
        full_results = []
        for algo in algos:
            random_action_scores = pb.read_baselines_from_files("random", capitalized_games, algo)
            trained_action_scores = pb.read_baselines_from_files("trained", capitalized_games, algo)
            print(f"random_action_scores for {algo}: {random_action_scores}")
            print(f"trained_action_scores for {algo}: {trained_action_scores}")

            all_results_for_algo = []
            for game_id, game in enumerate(games):
                print(f"{algo} {game}")
                env = f"{game[0].capitalize()}{game[1:].lower()}NoFrameskip-v4"
                rand_act_score, trained_act_score = random_action_scores[game_id], trained_action_scores[game_id]

                results_per_game = []
                for (percent, noise) in zip(noise_percents, all_noises):
                    result_mean = 0.
                    mode_repeats = 1 if sticky_action_prob == 0.0 else repeats
                    for nr in range(repeats):
                        results = perturbation_test.run_experiments_for_env(noise, env, algo=algo, sticky_action_prob=sticky_action_prob)
                        results = [results[0][0], results[1][0], results[2][0]]
                        print("No normalization", results)
                        normalized_result = utils.normalize(results, rand_act_score, trained_act_score)
                        print("After normalization", normalized_result)
                        result_mean += normalized_result

                    result_mean /= 5
                    results_per_game.append(result_mean)
                
                print("Results per game", results_per_game)
                all_results_for_algo.append(np.array(results_per_game))

            print("Results per algo", all_results_for_algo)

            full_results.append(all_results_for_algo)
            results_for_algo_array = np.array([np.array(per_game) for per_game in all_results_for_algo])
            mean_results_for_algo = np.mean(results_for_algo_array, axis=0)
            all_algos_results[algo] = mean_results_for_algo.tolist()

        print("Full results", full_results)
        print()
        print("Mean results", mean_results_for_algo)

def draw_sticky_plots(games):
    sticky_action_prob = 0.0
    noise_percents = [0.]
    noise_percents_temp = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    noise_percents.extend(noise_percents_temp)

    full_results = []
    all_algos_results = {}
    for algo in algos:
        all_results_for_algo = []
        for game_id, game in enumerate(games):
            path = f"sticky_results/{game}_{algo}_random_noise_{sticky_action_prob}.npy"            
            all_results_for_algo.append(np.load(path))
                
        full_results.append(all_results_for_algo)
        results_for_algo_array = np.array([np.array(per_game) for per_game in all_results_for_algo])
        mean_results_for_algo = np.mean(results_for_algo_array, axis=0)
        all_algos_results[algo] = mean_results_for_algo.tolist()

    plot_for_algos(all_algos_results, noise_percents, sticky_action_prob)

    for game_id, game in enumerate(games):
        game_plot_data = {}
        for i, algo_results in enumerate(full_results):
            game_plot_data[algos[i]] = algo_results[game_id]

        plot_for_algos(game_plot_data, noise_percents, sticky_action_prob, game=game)

def plot_for_algos(results, noise_percents, sticky_action_prob, game=None):
    sns.set_style('whitegrid')
    
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    plt.title(f'Test sticky_action_prob: {sticky_action_prob}', color='black')

    for i, (k, v) in enumerate(results.items()):
        idx = i % len(colors)
        plt.plot(noise_percents, v[:len(noise_percents)], colors[idx], label=k)

    plt.legend(loc='center left', bbox_to_anchor=(0.87,0.6))
    
    plt.xticks(noise_percents)
    plt.xticks(rotation=90)

    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Normalized Model Results")

    plt.ylim(0, 1.05)
    
    if game:
        plt.savefig(f'stinky_plots/plot_{game}_{sticky_action_prob}_test.png')
    else:
        plt.savefig(f'stinky_plots/plot_all_games_{sticky_action_prob}_0_test.png')

    plt.close()

    print("plot finished")


def algo_plot(results):
    sns.set_style('whitegrid')
    
    algos = ["rainbow", "dqn", "ga", "a2c", "impala", "es", "apex"]
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    plt.title(f'Sticky_action_prob comparison', color='black')
    labels = ["sticky_0_0", "sticky_0_25"]

    for i, v in enumerate(results):
        idx = i % len(colors)
        plt.plot(algos, v, colors[idx], label=labels[i])

    plt.legend(loc='center left', bbox_to_anchor=(0.80,0.6))
    
    plt.xticks(algos)

    plt.xlabel("Algorithms")
    plt.ylabel("Normalized Model Results")

    plt.ylim(0, 1.05)
    
    plt.savefig(f'stinky_plots/sticky_comparison.png')

    plt.close()


def normalized_color_table(data, path):
    plt.figure(figsize=(10, 10))
    plt.ion()
    plt.set_cmap('bwr')
    c = plt.pcolor(data) #, edgecolors='k', linewidths=4, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(c)
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    games = ["breakout", "seaquest", "enduro", "pong", "alien", "amidar", "gopher", "gravitar"]
    # plot_sticky_vs_non_sticky(games)

    # results = [
    #     [0.943, 0.870, 0.545, 0.720, 0.776, 0.738],
    #     [0.703, 0.776, 0.418, 0.461, 0.524, 0.630]
    # ]
    
    # algo_plot(results)

    # draw_sticky_plots(games)
    # get_results_for_no_perturbation_sticky(games)

    # make_lrs_plots(["breakout", "seaquest"])
    # data = np.random.random_sample((53, 53))
    # data[:, 12:17] = 0.7
    # normalized_color_table(data)
