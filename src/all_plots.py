import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2 as cv

import consts
import logs_reader
import utils
from consts import *

global_path = "./final_results" #_0_policy"

def get_games():
    return [
        "Alien",
        "Amidar",
        "Assault",
        "Asterix",
        "Asteroids",
        "Atlantis",
        "BankHeist",
        "BattleZone",
        "BeamRider",
        "Berzerk",
        "Bowling",
        "Boxing",
        "Breakout",
        "Centipede",
        "ChopperCommand",
        "CrazyClimber",
        "DemonAttack",
        "DoubleDunk",
        "Enduro",
        "FishingDerby",
        "Freeway",
        "Frostbite",
        "Gopher",
        "Gravitar",
        "Hero",
        "IceHockey",
        "Jamesbond",
        "Kangaroo",
        "Krull",
        "KungFuMaster",
        "MsPacman",
        "NameThisGame",
        "Phoenix",
        "Pong",
        "PrivateEye",
        "Qbert",
        "Riverraid",
        "RoadRunner",
        "Robotank",
        "Seaquest",
        "Skiing",
        "Solaris",
        "SpaceInvaders",
        "StarGunner",
        "Tennis",
        "TimePilot",
        "Tutankham",
        "UpNDown",
        "Venture",
        "VideoPinball",
        "WizardOfWor",
        "YarsRevenge",
        "Zaxxon"
    ]

def get_original_result(mode, game, algo, policy_id):
    path = f"./all_baselines_from_uber/{mode}/{game}_{algo}_policy_score.npy"
    results = np.load(path)

    return round(results[policy_id], 2)

def get_all_original_results_for_algo(games, algo, policy_id):
    all_results = {}
    for game in games:
        result = get_original_result("trained", game, algo, policy_id)
        all_results[game] = result

    return all_results

def get_all_random_actions_results_for_algo(games, algo, policy_id):
    all_results = {}
    for game in games:
        result = get_original_result("random", game, algo, policy_id)
        all_results[game] = result

    return all_results

def get_random_perturbation_result(mode, game, algo, run_id, policy_id, magnitude, given_path):
    for game in [game]:
        # print(os.getcwd())
        path = f"{given_path}/{mode}/{algo}/results/BankHeist/{run_id}/{magnitude}/{game.lower()}.npy"
        result = round(np.load(path)[0][policy_id], 2)
        # print(np.load(path))

    return result

def get_perturbation_result(mode, game, algo, run_id, policy_id, magnitude, given_path):
    for game in [game]:
        # print(os.getcwd())
        path = f"{given_path}/{mode}/{algo}/results/{game}/{run_id}/{magnitude}/{game.lower()}.npy"
        result = round(np.load(path)[0][policy_id], 2)
        # print(np.load(path))

    return result

def get_all_random_perturbations_average_results_for_algo(games, algo, policy_id, magnitude, path):
    all_results = {}
    for game in games:
        different_results_for_case = []
        for run_id in range(3):
            result = get_random_perturbation_result("random", game, algo, run_id, policy_id, magnitude, path)
            different_results_for_case.append(result)

        all_results[game] = np.mean(np.array(different_results_for_case))

    return all_results

def get_all_trained_perturbations_average_results_for_algo(games, algo, policy_id, magnitude, path):
    all_results = {}
    for game in games:
        different_results_for_case = []
        for run_id in range(3):
            result = get_perturbation_result("trained", game, algo, run_id, policy_id, magnitude, path)
            different_results_for_case.append(result)

        all_results[game] = np.mean(np.array(different_results_for_case))

    return all_results

def convert_to_mean_results_for_games(results):
    algo_results = []
    for game_results in results:
        converted_results = {}
        for k, v in game_results.items():
            converted_results[k] = np.mean(np.array(v))

        algo_results.append(converted_results)
    
    return algo_results

def convert_to_mean_results_for_algo(results):
    nr_magnitudes = len(results[0].items())
    results = convert_to_mean_results_for_games(results)
    algo_means = {k: 0.0 for k,v in results[0].items()}
    for game_results in results:
        for k, v in game_results.items():
            algo_means[k] += v
    
    return {k: v / nr_magnitudes for k, v in algo_means.items()}

def get_results_with_trained_perturbations(games, results):
    peturbation_trained_results = []
    for i in range(len(games)):
        for j in range(len(results)):
            if i == j:
                peturbation_trained_results.append(results[i])

    return peturbation_trained_results

def get_results_only_for_trained_pert(games, results):
    peturbation_trained_results = []
    for i in range(len(games)):
        one_game_results = {}
        for k, v in results[i].items():
            for j, val in enumerate(v):
                if i == j:
                    one_game_results[k] = val
            
        peturbation_trained_results.append(one_game_results)

    return peturbation_trained_results

def get_results_for_one_policy(log_path=None, algo=None, mode=None, policy_id=0, games=None, max_noises=[]):
    path = f"{log_path}/{mode}"
    print("AAAAA", path)
    # print(path, algo, max_noises, games, policy_id)
    
    if mode == "trained":
        results, mini, maxi, std, _ = logs_reader.get_results_for_trained_games(path, algo, max_noises=max_noises, log_games=games, policy_id=policy_id)
    elif mode == "random":
        print("AAA")
        results, mini, maxi, std, _ = logs_reader.get_results_for_random_games(algo, max_noises=max_noises, log_games=games, policy_id=policy_id)
    
    return results, mini, maxi, std

def get_mean_result(results, max_noises):
    mean_results = {}
    len_all = len(results)
    for noise in max_noises:
        noise_res = 0.0
        for i, res in enumerate(results):
            noise_res += res[noise]
        noise_res = noise_res / float(len_all)
        mean_results[noise] = noise_res
    
    return mean_results

def get_mean_result_for_all_games(results, max_noises):
    mean_results = {}
    len_all = len(results)
    for noise in max_noises:
        noise_res = 0.0
        for i, res in enumerate(results):
            noise_res += np.mean(res[noise])
        noise_res = noise_res / float(len_all)
        mean_results[noise] = noise_res
    
    return mean_results

def get_stddevs(results, max_noises):
    std_results = {}

    len_all = len(results)
    for noise in max_noises:
        all_results_for_one_noise = [res[noise] for i, res in enumerate(results)]
        # for res in all_results_for_one_noise:
        #     print(res)
        std_results[noise] = np.std(np.array(all_results_for_one_noise))

    return std_results

def get_mean_result_for_few_policies(log_path=None, algo=None, mode=None, policy_id=[0], games=None, max_noises=[]):
    results0, min_results0, max_results0, std0 = get_results_for_one_policy(log_path=global_path, algo=algo, policy_id=0, games=games, mode=mode, max_noises=max_noises)
    results1, min_results1, max_results1, std1 = get_results_for_one_policy(log_path=global_path, algo=algo, policy_id=1, games=games, mode=mode, max_noises=max_noises)
    results2, min_results2, max_results2, std2 = get_results_for_one_policy(log_path=global_path, algo=algo, policy_id=2, games=games, mode=mode, max_noises=max_noises)

    mean_results0 = get_mean_result(results0, max_noises)
    mean_results1 = get_mean_result(results1, max_noises)
    mean_results2 = get_mean_result(results2, max_noises)
    
    min_results0 = get_mean_result(min_results0, max_noises)
    min_results1 = get_mean_result(min_results1, max_noises)
    min_results2 = get_mean_result(min_results2, max_noises)   

    max_results0 = get_mean_result(max_results0, max_noises)
    max_results1 = get_mean_result(max_results1, max_noises)
    max_results2 = get_mean_result(max_results2, max_noises)

    averaged_std_0 = average_std(std0, max_noises)
    averaged_std_1 = average_std(std1, max_noises)
    averaged_std_2 = average_std(std2, max_noises)

    mean_results = [mean_results0, mean_results1, mean_results2]
    min_results = [min_results0, min_results1, min_results2]
    max_results = [max_results0, max_results1, max_results2]
    averaged_std_results = [averaged_std_0, averaged_std_1, averaged_std_2]
    
    final_mean_results, final_min_results, final_max_results, final_std_results = {}, {}, {}, {}
    for noise in max_noises:
        result, min_result, max_result, std_result, cnt = 0.0, 0.0, 0.0, 0.0, 0
        for i in range(3):
            if i in policy_id:
                cnt += 1
                result += mean_results[i][noise]
                min_result += min_results[i][noise]
                max_result += max_results[i][noise]
                std_result += averaged_std_results[i][noise]

        result = result / cnt
        min_result = min_result / cnt
        max_result = max_result / cnt
        std_result = std_result / cnt

        final_mean_results[noise] = result
        final_min_results[noise] = min_result
        final_max_results[noise] = max_result
        final_std_results[noise] = std_result

    return final_mean_results, final_min_results, final_max_results, final_std_results

def plot_mean_results(results, percents, path):
    sns.set_style('whitegrid')
    percents2 = [percent.replace('_', '.') for percent in percents]
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    plt.title(f"Average normalized for {results[0][0]}", color='black')
    modes = ["random", "trained"]
    for cnt, (algo, algo_results) in enumerate(results):
        vs = [algo_results[percent] for percent in percents]
        plt.plot(percents2, vs, colors[cnt % len(colors)], label=modes[cnt])

    plt.legend(loc='center left', bbox_to_anchor=(0.75,0.8))
    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Normalized Expected Return")
    # plt.xscale('log')

    plt.ylim(0, 1.05)
    plt.savefig(path)
    plt.close()

def plot_mean_results_with_bounds(results, min_results, max_results, percents, path):
    sns.set_style('whitegrid')
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']

    fig, ax = plt.subplots()
    percents2 = [p.replace("_", ".") for p in percents]
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

def plot_mean_results_other_algo(results, percents, path):
    sns.set_style('whitegrid')
    percents2 = [percent.replace('_', '.') for percent in percents]
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']
    plt.title(f"Average normalized for {results[0][0]}", color='black')
    algos = ["Random perturbation", "Trained DQN", "Trained Rainbow"]

    for cnt, (algo, algo_results) in enumerate(results):
        vs = [algo_results[percent] for percent in percents]
        plt.plot(percents2, vs, colors[cnt % len(colors)], label=algos[cnt])

    plt.legend(loc='center left', bbox_to_anchor=(0.6, 0.9))
    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Normalized Expected Return")
    # plt.xscale('log')

    plt.ylim(0, 1.05)
    plt.savefig(path)
    plt.close()

def plot_mean_results_with_bounds_other_algo(results, min_results, max_results, percents, path, algo):
    sns.set_style('whitegrid')
    colors = ['orange', 'blue', 'brown', 'green', 'red', 'purple', 'black', 'yellow', 'gray', 'pink', 'olive','darkblue']

    fig, ax = plt.subplots()
    percents2 = [p.replace("_", ".") for p in percents]
    algos = ["Random perturbation", f"Trained perturbation on {algo.upper()}", "Trained perturbation on Rainbow"]
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

    x, y = np.arange(20), np.arange(20)
    games = utils.get_sampled_games()
    plt.rcParams["axes.grid"] = False

    ax.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(games, fontsize=8, rotation=50)  
    ax.set_yticks(y)
    ax.set_yticklabels(games, fontsize=8, rotation=50)
    # # create an axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.20)

    plt.colorbar(im, cax=cax)

    # plt.colorbar(im)
    plt.savefig(path)
    plt.close()

def create_transfer_table(results=None, path=None, games=None):
    # transfer_power = get_transfer_power(results)
    # utils.fix_path()
    if results is None:
        colormap = np.arange(53*53).reshape(53, 53)
        colormap = colormap / 2809
    else:
        colormap = results

    plot_color_table(colormap, path, games)
    # cv2.imwrite("test.png", colormap*256)
    # return "test.png", transfer_power

def remove_cross_from_table(transfer_table):
    transfer_power_without_cross = np.copy(transfer_table)
    for i in range(transfer_power_without_cross.shape[0]):
        transfer_power_without_cross[i,i] = 0.0
    
    return transfer_power_without_cross

def average_std(std_results, max_noises):
    averaged_std = {}
    for max_noise in max_noises:
        avg = np.mean(np.array([result[max_noise] for result in std_results]))
        averaged_std[max_noise] = avg
        
    return averaged_std

def draw_mean_plot_for_one_policy(algo, max_noises, games, policy_id, path):
    # Usrednione wyniki dla wybranej polityki i jednego kilku magnitude pomiedzy wszystkimi grami
    all_results, left_std_results, right_std_results = [], [], []
    modes = ["random", "trained"]
    paths = ["./final_results_0_policy", "./final_results"]

    for ii, mode in enumerate(modes):
        mean_results, min_results, max_results, std_results = get_results_for_one_policy(
            log_path=paths[ii], algo=algo, policy_id=policy_id, games=games, mode=mode, max_noises=max_noises)

        all_results.append((algo, mean_results))

        left_bounds_with_stddev = {noise: max([mean_results[noise] - std_results[noise], 0]) for noise in max_noises}
        left_std_results.append((algo, left_bounds_with_stddev))
        
        right_bounds_with_stddev = {noise: min([mean_results[noise] + std_results[noise], 1]) for noise in max_noises}
        right_std_results.append((algo, right_bounds_with_stddev))

    plot_mean_results(all_results, max_noises, f'final_plots/one_policy_{algo}_{policy_id}.png')
    plot_mean_results_with_bounds(all_results, left_std_results, right_std_results, max_noises, f'final_plots/one_policy_stddev_{algo}_{policy_id}.png')

def draw_mean_plot_for_one_policy_and_multi_game(algos, max_noises, games, policy_id, log_path, mode):
    # Usrednione wyniki dla wybranej polityki i jednego kilku magnitude pomiedzy wszystkimi grami
    # all_results, all_min_results, all_max_results, left_std_results, right_std_results = [], [], [], [], []
    all_results, all_min_results, all_max_results  = [], [], []
    for algo in algos:
        path = f"{log_path}/{mode}"

        results, min_results, max_results, std_results, _ = logs_reader.get_all_results(path, algo, max_noises=max_noises, 
            log_games=games, policy_id=policy_id)

        mean_results = get_mean_result_for_all_games(results, max_noises)
        all_results.append((algo, mean_results))

        mean_min_results = get_mean_result_for_all_games(min_results, max_noises)
        all_min_results.append((algo, mean_min_results))

        mean_max_results = get_mean_result_for_all_games(max_results, max_noises)
        all_max_results.append((algo, mean_max_results))

    plot_mean_results(all_results, max_noises, f'final_plots/one_policy_multi_game_{mode}.png')
    plot_mean_results_with_bounds(all_results, all_min_results, all_max_results, max_noises, 
        f'final_plots/one_policy_multi_game_minmax_{mode}.png')

def draw_mean_plot_for_few_policies(algos, max_noises, games, policies, path, mode):
    all_results, all_min_results, all_max_results, all_left_std, all_right_std = [], [], [], [], []
    algo = algos[0]
    modes = ["random", "trained"]
    for mode in modes:
        mean_results, min_results, max_results, std_results = get_mean_result_for_few_policies(log_path=global_path, algo=algo, 
                                                        policy_id=policies, games=games, mode=mode, max_noises=max_noises)

        all_results.append((algo, mean_results))
        all_min_results.append((algo, min_results))
        all_max_results.append((algo, max_results))
            
        left_stds = {noise: max([mean_results[noise] - std_results[noise], 0]) for noise in max_noises}
        all_left_std.append((algo, left_stds))
        
        right_stds = {noise: min([mean_results[noise] + std_results[noise], 1]) for noise in max_noises}
        all_right_std.append((algo, right_stds))

    plot_mean_results(all_results, max_noises, f'final_plots/few_policy_{algo}.png')
    plot_mean_results_with_bounds(all_results, all_min_results, all_max_results, max_noises, f'final_plots/few_policy_minmax_{algo}.png')
    plot_mean_results_with_bounds(all_results, all_min_results, all_left_std, all_right_std, f'final_plots/few_policy_stddev_{algo}.png')

def draw_mean_plot_for_few_policies_and_multi_game(algos, max_noises, games, policies, path, mode):
    all_results, all_min_results, all_max_results = [], [], []
    for algo in algos:
        mean_results, min_results, max_results = get_mean_result_for_few_policies_and_multi_game(log_path=global_path, algo=algo, 
                                                        policy_id=policies, games=games, mode=mode, max_noises=max_noises)

        all_results.append((algo, mean_results))
        all_min_results.append((algo, min_results))
        all_max_results.append((algo, max_results))

    plot_mean_results(all_results, max_noises, f'final_plots/few_policy_multi_game_{mode}.png')
    plot_mean_results_with_bounds(all_results, all_min_results, all_max_results, max_noises, f'final_plots/few_policy_minmax_multi_game_{mode}.png')

def draw_transfer_table_2(algos, max_noises, policy_id, mode, games, global_path):
    all_results, all_min_results, all_max_results, left_std_results, right_std_results = [], [], [], [], []
    for algo in algos:
        print(algo)
        path = f"{global_path}/{mode}"

        # poprawic buga z policies (obecnie dobrze dziala jedynie dla pojedynczego policy_id, lista tez potrzebna)
        results = logs_reader.get_results_for_transfer_table(path, algo, max_noises=max_noises, policy_id=policy_id, games=games)

        for max_noise, transfer_table_for_noise in results.items():
            print(max_noise)
            transfer_table = np.array(transfer_table_for_noise)
            transfer_power = np.mean(transfer_table)
            
            transfer_table_without_cross = remove_cross_from_table(transfer_table)
            shape_without_cross = transfer_table_without_cross.shape[0]
            transfer_power_without_cross = np.sum(transfer_table_without_cross) / (shape_without_cross * (shape_without_cross-1))
            
            print(transfer_power, transfer_power_without_cross)

            path_to_save_transfer_table = f"./final_plots/transfer_tables/{algo}_{max_noise}_{mode}.png"
            create_transfer_table(results=transfer_table, path=path_to_save_transfer_table, games=games)

            path_to_save_transfer_power = f"./final_plots/transfer_powers/{algo}_{max_noise}_{mode}.npy"
            open(path_to_save_transfer_power, 'a').close()
            np.save(path_to_save_transfer_power, np.array([transfer_power]))

            path_to_save_transfer_power = f"./final_plots/transfer_powers_without_cross/{algo}_{max_noise}_{mode}.npy"
            open(path_to_save_transfer_power, 'a').close()
            np.save(path_to_save_transfer_power, np.array([transfer_power_without_cross]))

def open_random_pert(paths):
    for path in paths:
        print(path)
        a = np.load(f"final_results/random/random_perts/{path}")
        parsed_max_noise = path.split("_")[2]
        max_noise_float = float(f"0.{parsed_max_noise}")
        print(max_noise_float)
        print(np.max(a))
        print(np.min(a))
        if np.max(a) > max_noise_float:
            return  False
        if np.min(a) < -max_noise_float:
            return  False
    
    return True

def check_if_case_completed(path, all_exps_len=20):
    max_noises = os.listdir(path)
    if len(max_noises) == 1:
        print("All logs missing!")
        return False, 1
    
    # max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    max_noises  = ["0_1"]
    for max_noise in max_noises:        
        noise_path = f"{path}/{max_noise}"
        if not os.path.exists(noise_path):
            print("Not exists", noise_path)
            continue

        experiments = os.listdir(noise_path)

        if (len(experiments) < all_exps_len) or (len(experiments) > all_exps_len):
            print(f"Completed only {len(experiments)} < {all_exps_len} games", max_noise)
            return False, 2
    
    return True, 0

def completion_check_for_all_games(algos, mode, policy_id, main_path):
    games = utils.get_sampled_games() # full_dopamine_games_list

    for algo in algos:
        print(algo)
        missing_games = []
        partially_missing_games = []
        for game in games:
            path = f"{main_path}/{mode}/{algo}/results/{game}/{policy_id}"
            status, case = check_if_case_completed(path, all_exps_len=len(games))

            if status == False and case == 1:
                missing_games.append(game)
            
            if status == False and case == 2:
                partially_missing_games.append(game)
                
        print(missing_games)
        print(partially_missing_games)

### Main functions ###

def zero_policy_completion():
    games = utils.get_sampled_games()
    main_path = "./final_results_0_policy"
    algos = ["rainbow"] # ["dqn", "a2c", "ga", "es"]
    policy_id = 0
    mode = "trained"
    max_noises = ["0_005", "0_008", "0_01", "0_05", "0_1"]

    for algo in algos:
        missing_games = []
        partially_missing_games = []
        for game in games:
            path = f"{main_path}/{mode}/{algo}/results/{game}/{policy_id}"
            
            for max_noise in max_noises:
                magnitude_path = f"{path}/{max_noise}"
                games_path = os.listdir(magnitude_path)

                cnt = 0
                for game_path in games_path:
                    final_path = f"{magnitude_path}/{game_path}"
                    # print(final_path)

                    result = np.load(final_path)
                    # print(result)
                    if result.shape[0] == 1 and result[0].shape[0] == 1:
                        cnt += 1

                if cnt < 20:
                    print(magnitude_path)

def if_logs_completed():
    # algos = ["a2c", "dqn", "es", "ga", "rainbow"]
    path = "./final_results"#_0_policy"
    algos = ["rainbow"]
    policy_id = 0
    nr_perts = 3
    for policy_id in range(nr_perts):
        # mode = "random"
        # completion_check_for_all_games(algos, mode, policy_id, path)
        
        mode = "trained"
        completion_check_for_all_games(algos, mode, policy_id, path)
        print()

def get_all_results_for_one_policy():
    # max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    max_noises = ["0_01"]
    # algos = ["a2c", "dqn", "es", "ga", "rainbow"]
    algos = ["ga"]
    all_games = sorted(utils.get_sampled_games())

    for game in all_games:
        print("Current game:", game)

        games = [game]
        policy_id = 0
        path = "./final_results_0_policy"

        for algo in algos:
            print(f"Algorithm {algo}")
            for magnitude in max_noises:
                print(f"Magnitude {magnitude}")
            
                original = get_all_original_results_for_algo(games, algo, policy_id)
                print("Original", original)

                random_pert = get_all_random_perturbations_average_results_for_algo(games, algo, policy_id, magnitude, path)
                # random_pert = get_all_random_perturbations_results_for_algo(games, algo, policy_id, magnitude, path)
                print("Random perturbations", random_pert)

                trained_pert = get_all_trained_perturbations_average_results_for_algo(games, algo, policy_id, magnitude, path)
                # trained_pert = get_all_trained_perturbations_results_for_algo(games, algo, policy_id, magnitude, path)
                print("Trained perturbations", trained_pert)

                rand_act = get_all_random_actions_results_for_algo(games, algo, policy_id)
                print("Random actions", rand_act)

                print()
        
def draw_averaged_plots():
    max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    algos = ["dqn", "es", "ga", "rainbow", "a2c"]
    games = utils.get_sampled_games()
    path = "./final_results"#_0_policy"
    policy_id = 0
    for algo in algos:
        draw_mean_plot_for_one_policy(algo, max_noises, games, policy_id, path)

def draw_transfer_table():
    max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    algos = ["a2c", "dqn", "es", "ga", "rainbow"]
    games = utils.get_sampled_games()

    policy_id = 0
    mode = "trained"
    algos = ["ga"]
    draw_transfer_table_2(algos, max_noises, policy_id, mode, games, "./final_results_0_policy")

def get_random_transfer_performance_drop():
    algo = "es"
    games = utils.get_sampled_games()
    max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    policy_id = 0

    logs_reader.get_transfer_power_for_random_results(algo, max_noises, games, policy_id)

def get_trained_transfer_performance_drop():
    algo = "es"
    games = utils.get_sampled_games()
    max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
    policy_id = 0
    mode = "trained"
    path = f"./final_results_0_policy/{mode}"

    results = logs_reader.get_results_for_transfer_table(path, algo, max_noises=max_noises, policy_id=policy_id, games=games)

    for max_noise, transfer_table_for_noise in results.items():
        transfer_table = np.array(transfer_table_for_noise)
        transfer_power = np.mean(transfer_table)

        # print(f"{max_noise} with cross => {transfer_power}")    
        
        transfer_table_without_cross = remove_cross_from_table(transfer_table)
        shape_without_cross = transfer_table_without_cross.shape[0]
        transfer_power_without_cross = np.sum(transfer_table_without_cross) / (shape_without_cross * (shape_without_cross-1))
        
        print(f"{max_noise} without cross => {round(transfer_power_without_cross,4)}")    

def merge_images():
    path1 = "./final_plots/stddev_up.png"
    path2 = "./final_plots/stddev_down.png"
   
    img1, img2 = cv.imread(path1, 1), cv.imread(path2, 1)
    print(img1.shape)
    print(img2.shape)
    figure = np.zeros((2*img1.shape[0], img1.shape[1], img1.shape[2]))
    figure[:img1.shape[0], :, :] = img1
    diff_one_side = int((img1.shape[1] - img2.shape[1])/2)
    figure[img1.shape[0]:, diff_one_side: img1.shape[1]-diff_one_side, :] = img2
    
    figure[img1.shape[0]:, :diff_one_side, :] = 255
    figure[img1.shape[0]:, img1.shape[1]-diff_one_side:, :] = 255
    
    print(figure[img1.shape[0]:, img1.shape[1]-diff_one_side:, :].shape)
    print(np.min(img1))
    cv.imwrite("final_plots/train_1_test_0_stddev.png", figure)

def draw_mean_plot_for_one_policy_for_algo(algo, max_noises, games, policy_id, path):
    # Usrednione wyniki dla wybranej polityki i jednego kilku magnitude pomiedzy wszystkimi grami
    all_results, left_std_results, right_std_results = [], [], []
    modes = ["random", "trained"]
    paths = ["./final_results_0_policy", "./final_results"]

    for ii, mode in enumerate(modes):
        mean_results, min_results, max_results, std_results = get_results_for_one_policy(
            log_path=paths[ii], algo=algo, policy_id=policy_id, games=games, mode=mode, max_noises=max_noises)

        all_results.append((algo, mean_results))

        left_bounds_with_stddev = {noise: max([mean_results[noise] - std_results[noise], 0]) for noise in max_noises}
        left_std_results.append((algo, left_bounds_with_stddev))
        
        right_bounds_with_stddev = {noise: min([mean_results[noise] + std_results[noise], 1]) for noise in max_noises}
        right_std_results.append((algo, right_bounds_with_stddev))

    plot_mean_results(all_results, max_noises, f'final_plots/one_policy_{algo}_{policy_id}.png')
    plot_mean_results_with_bounds(all_results, left_std_results, right_std_results, max_noises, f'final_plots/one_policy_stddev_{algo}_{policy_id}.png')

def plot_for_other_algo():
    algos = ["dqn", "a2c", "ga", "es"]
    for algo in algos:
        main_algo_path = "./final_results_0_policy"
        other_algos_path = "./final_results_with_rainbow_pert"
        
        paths = [main_algo_path, main_algo_path, other_algos_path]
        modes = ["random", "trained", "trained"]
        algos = [algo, algo, algo]
        games = utils.get_sampled_games()
        max_noises = ["0_001", "0_005", "0_008", "0_01", "0_05", "0_1"]
        policy_id = 0

        all_results, left_std_results, right_std_results = [], [], []
        for ii, algo in enumerate(algos):
            print(paths[ii])
            mean_results, min_results, max_results, std_results = get_results_for_one_policy(
                log_path=paths[ii], algo=algo, policy_id=policy_id, games=games, mode=modes[ii], max_noises=max_noises)

            all_results.append((algo, mean_results))

            left_bounds_with_stddev = {noise: max([mean_results[noise] - std_results[noise], 0]) for noise in max_noises}
            left_std_results.append((algo, left_bounds_with_stddev))
            
            right_bounds_with_stddev = {noise: min([mean_results[noise] + std_results[noise], 1]) for noise in max_noises}
            right_std_results.append((algo, right_bounds_with_stddev))

        plot_mean_results_other_algo(all_results, max_noises, f'final_plots/other_algo_{algo}_{policy_id}.png')
        plot_mean_results_with_bounds_other_algo(all_results, left_std_results, right_std_results, max_noises, f'final_plots/other_algo_stddev_rainbow_{algo}_{policy_id}.png', algo)

if __name__ == "__main__":
    # zero_policy_completion()

    # if_logs_completed()

    # get_all_results_for_one_policy()
    
    draw_averaged_plots()

    # draw_transfer_table()

    # get_random_transfer_performance_drop()
    
    # get_trained_transfer_performance_drop()

    # merge_images()

    # plot_for_other_algo()