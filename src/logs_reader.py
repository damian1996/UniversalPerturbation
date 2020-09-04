import numpy as np
import sys
import os

import transfer_plots as t_plots
import utils


def game_name_converter(curr_name, correct_names):
    for corr_name in correct_names:
        if curr_name.lower() == corr_name.lower():
            return corr_name

def get_results_for_transfer_table(path, algo, max_noises=None, policy_id=None, games=None):
    all_baselines_trained_path = "./all_baselines_from_uber/trained"
    all_baselines_random_path = "./all_baselines_from_uber/random"

    desired_games = games if games is not None else sorted(utils.get_games_for_algo(algo))
    print(desired_games)

    for algo in [algo]:
        for category in ["results"]:
            trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
            random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}

            all_results_for_plot = {noise: np.zeros((len(desired_games), len(desired_games))) for noise in max_noises if noise.startswith("0_")}

            for game_id, game in enumerate(desired_games):
                nr_perts = [0, 1, 2]

                for nr_pert in nr_perts:
                    for max_noise in max_noises:
                        test_case_path = f'{path}/{algo}/{category}/{game}/{nr_pert}/{max_noise}'

                        for test_game_id, test_game in enumerate(desired_games):
                            result = np.load(f"{test_case_path}/{test_game.lower()}.npy")
                            trained_act_score = trained_dict[test_game]
                            random_act_score = random_dict[test_game]

                            result = [[result[0][policy_id]]]
                            result = utils.normalize_result_for_one_policy(result, random_act_score[policy_id], trained_act_score[policy_id], test_game)

                            all_results_for_plot[max_noise][game_id][test_game_id] += result
                
            nr_perts_len = len(nr_perts)
            for max_noise in max_noises:
                for game_id in range(len(desired_games)):
                    for test_game_id in range(len(desired_games)):
                        val = all_results_for_plot[max_noise][game_id][test_game_id]
                        all_results_for_plot[max_noise][game_id][test_game_id] = val / nr_perts_len
                
    return all_results_for_plot

def get_transfer_power_for_random_results(algo, max_noises=None, log_games=None, policy_id=None):
    path = "./final_results_0_policy/random"

    all_baselines_trained_path = "./all_baselines_from_uber/trained"
    all_baselines_random_path = "./all_baselines_from_uber/random"

    desired_games = sorted(log_games)

    trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
    random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}

    for max_noise in max_noises:
        cnt, summarized_result = 0, 0.0
        for game in desired_games:
            trained_act_score = trained_dict[game]
            random_act_score = random_dict[game]
            for nr_pert in range(3):
                case_path = f"./final_results_0_policy/random/{algo}/results/BankHeist/{nr_pert}/{max_noise}/{game.lower()}.npy"       
                result = [[np.load(case_path)[0][policy_id]]]
                result = utils.normalize_result_for_one_policy(result, random_act_score[policy_id], trained_act_score[policy_id], game)
                
                summarized_result += result
                cnt += 1

        mean_result = summarized_result / cnt
        print(f"{max_noise} => {round(mean_result, 4)}")


def get_results_for_trained_games(path, algo, nr_perts_param=None, max_noises=None, log_games=None, policy_id=None):
    categories = ["results"]

    all_baselines_trained_path = "./all_baselines_from_uber/trained"
    all_baselines_random_path = "./all_baselines_from_uber/random"

    if log_games:
        desired_games = sorted(log_games)
    else:
        desired_games = sorted(utils.get_games_for_algo(algo))

    for algo in [algo]:
        for category in categories:
            trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
            random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}

            nr_perts = [0, 1, 2]

            game_results = {noise: 0.0 for noise in max_noises if noise.startswith("0_")}
            min_results = {noise: 1000000.0 for noise in max_noises if noise.startswith("0_")}
            max_results = {noise: -1000000.0 for noise in max_noises if noise.startswith("0_")}
            all_results_for_plot = {noise: [] for noise in max_noises if noise.startswith("0_")}
            std_results = {noise: 0.0 for noise in max_noises if noise.startswith("0_")}

            for nr_pert in nr_perts:
                for max_noise in max_noises:
                    summarized_result = 0.0
                    for game in desired_games:
                        test_case_path = f'{path}/{algo}/{category}/{game}/{nr_pert}/{max_noise}'
                             
                        result = np.load(f"{test_case_path}/{game.lower()}.npy")
                        trained_act_score = trained_dict[game]
                        random_act_score = random_dict[game]
                        # print(f"{test_case_path}/{game.lower()}.npy", result)
                        result = [[result[0][policy_id]]]
                        result = utils.normalize_result_for_one_policy(result, random_act_score[policy_id], trained_act_score[policy_id], game)
                        summarized_result += result

                    mean_result = summarized_result / len(desired_games)
                    game_results[max_noise] += mean_result
                    min_results[max_noise] = min(min_results[max_noise], mean_result)
                    max_results[max_noise] = max(max_results[max_noise], mean_result)
                    all_results_for_plot[max_noise].append(mean_result)
                        
            nr_perts_len = len(nr_perts)
            for max_noise, v in game_results.items():
                game_results[max_noise] =  v / nr_perts_len
                
            for max_noise in max_noises:
                std_results[max_noise] = np.std(all_results_for_plot[max_noise])
                            
    return game_results, min_results, max_results, std_results, []

def get_results_for_random_games(algo, nr_perts_param=None, max_noises=None, log_games=None, policy_id=None):
    category = "results"
    path = "./final_results_0_policy/random"

    all_baselines_trained_path = "./all_baselines_from_uber/trained"
    all_baselines_random_path = "./all_baselines_from_uber/random"

    if log_games:
        desired_games = sorted(log_games)
    else:
        desired_games = sorted(utils.get_games_for_algo(algo))

    for algo in [algo]:
        trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
        random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}

        nr_perts = [0, 1, 2]
        ultimate_game = "BankHeist"
        
        game_results = {noise: 0.0 for noise in max_noises if noise.startswith("0_")}
        all_results_for_plot = {noise: [] for noise in max_noises if noise.startswith("0_")}
        std_results = {noise: 0.0 for noise in max_noises if noise.startswith("0_")}

        for nr_pert in nr_perts:
            for max_noise in max_noises:
                summarized_result = 0.0
                for game_id, game in enumerate(desired_games):
                    test_case_path = f'{path}/{algo}/{category}/{ultimate_game}/{nr_pert}/{max_noise}'
                    result = np.load(f"{test_case_path}/{game.lower()}.npy")
                    trained_act_score = trained_dict[game]
                    random_act_score = random_dict[game]

                    result = [[result[0][policy_id]]]
                    # print(random_act_score[policy_id], result, trained_act_score[policy_id])
                    result = utils.normalize_result_for_one_policy(result, random_act_score[policy_id], trained_act_score[policy_id], game)
                    summarized_result += result

                mean_result = summarized_result / len(desired_games)
                game_results[max_noise] += mean_result
                all_results_for_plot[max_noise].append(mean_result)

        nr_perts_len = len(nr_perts)
        for max_noise, v in game_results.items():
            game_results[max_noise] =  v / nr_perts_len
           
        for max_noise in max_noises:
            std_results[max_noise] = np.std(all_results_for_plot[max_noise])
                
    return game_results, [], [], std_results, []

def get_all_results(path, algo, nr_perts_param=None, max_noises=None, log_games=None, policy_id=None):
    categories = ["results", "normalized_results", "perts"]

    all_baselines_trained_path = "./all_baselines_from_uber/trained"
    all_baselines_random_path = "./all_baselines_from_uber/random"

    if log_games:
        desired_games = sorted(log_games)
    else:
        desired_games = sorted(utils.get_games_for_algo(algo))

    for algo in [algo]:
        for category in categories:
            if category != "results":
                continue
            
            sumi, cnt = 0.0, 0

            all_perts_games = sorted(os.listdir(f'{path}/{algo}/{category}'))

            trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
            random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}

            all_games_results = []
            all_min_results = []
            all_max_results = []
            all_std_results = []

            for game in desired_games:
                # print(game)
                nr_perts = sorted(os.listdir(f'{path}/{algo}/{category}/{game}'))
                nr_perts = nr_perts_param if nr_perts_param else [0, 1, 2]

                for nr_pert in nr_perts:
                    if not max_noises:
                        max_noises = os.listdir(f'{path}/{algo}/{category}/{game}/{nr_pert}')

                    if int(nr_pert) == nr_perts[0]:
                        # print(max_noises)
                        game_results = {noise: np.zeros((len(desired_games), )) for noise in max_noises if noise.startswith("0_")}
                        min_results = {noise: np.array([1000000.0] * len(desired_games)) for noise in max_noises if noise.startswith("0_")}
                        max_results = {noise: np.array([-1000000.0] * len(desired_games)) for noise in max_noises if noise.startswith("0_")}
                        max_noise_fully_filled = {noise: 0 for noise in max_noises if noise.startswith("0_")}
                        all_results_for_plot = {noise: [[]] * len(desired_games) for noise in max_noises if noise.startswith("0_")}
                        std_results = {noise: np.zeros((len(desired_games), )) for noise in max_noises if noise.startswith("0_")}
                        # print(all_results_for_plot.items())

                    if len(max_noises) > 0:
                        initial_results = np.array([0.0 for game in desired_games])
                        
                        for max_noise in max_noises:
                            if ".py" in max_noise:
                                continue
                            
                            if max_noise not in game_results:
                                continue
                            
                            test_case_path = f'{path}/{algo}/{category}/{game}/{nr_pert}/{max_noise}'
                            desired_games_logs = [f'{game.lower()}.npy' for game in desired_games]
                            
                            for g_id, res_game in enumerate(desired_games_logs):
                                result = np.load(f"{test_case_path}/{res_game}")
                                tested_game = desired_games[g_id] #res_game.split('.')[0][0].capitalize() + res_game.split('.')[0][1:]
                                tested_game = game_name_converter(tested_game, all_perts_games)

                                trained_act_score = trained_dict[tested_game]
                                random_act_score = random_dict[tested_game]
                                if policy_id:
                                    result = [[result[0][policy_id]]]
                                    result = utils.normalize_result_for_one_policy(result, random_act_score, trained_act_score, tested_game)
                                else:
                                    result = utils.normalize2(result, random_act_score, trained_act_score, tested_game)
                                
                                game_results[max_noise][g_id] += result     
                                min_results[max_noise][g_id] = min(min_results[max_noise][g_id], result)
                                max_results[max_noise][g_id] = max(max_results[max_noise][g_id], result)
                                all_results_for_plot[max_noise][g_id].append(result)
                                
                            max_noise_fully_filled[max_noise] += 1        

                for max_noise, v in game_results.items():
                    game_results[max_noise] =  v / max_noise_fully_filled[max_noise]
                
                for max_noise in max_noises:
                    for g_id in desired_games_logs:
                        std_results[max_noise][g_id] = np.std(all_results_for_plot[max_noise][g_id])
                    
                if len(game_results.items()) > 0:
                    all_games_results.append(game_results)
                    all_min_results.append(min_results)
                    all_max_results.append(max_results)
                    all_std_results.append(std_results)

    # return all_games_results, all_min_results, all_max_results, all_perts_games
    return all_games_results, all_min_results, all_max_results, all_std_results, all_perts_games

def get_normalized_result(path, algo):
    results, _ = get_all_results(path, algo)
    all_mean_results = []
    noises = list(results[0].keys())

    for noise in noises:
        full_mean = np.mean(np.array([np.mean(result[noise]) for result in results]))
        all_mean_results.append(full_mean)

    return all_mean_results

def get_transfer_table(path, algo):
    results, games = get_all_results(path, algo)
    
    all_mean_results = []
    noises = list(results[0].keys())
    for noise in noises:
        full_table = [result[noise] for result in results]
        all_mean_results.append(np.array(full_table))

    return all_mean_results

def get_original_results(algo, game):
    path = "./all_baselines_from_uber/trained"
    print(np.load(f"{path}/{game}_{algo}_policy_score.npy"))


# def get_results_for_trained_games(path, algo, nr_perts_param=None, max_noises=None, log_games=None, policy_id=None):
#     categories = ["results", "normalized_results", "perts"]
#     categories = ["results"]

#     all_baselines_trained_path = "./all_baselines_from_uber/trained"
#     all_baselines_random_path = "./all_baselines_from_uber/random"

#     if log_games:
#         desired_games = sorted(log_games)
#     else:
#         desired_games = sorted(utils.get_games_for_algo(algo))

#     for algo in [algo]:
#         for category in categories:
#             all_perts_games = sorted(os.listdir(f'{path}/{algo}/{category}'))

#             trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
#             random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}

#             all_games_results = []
#             all_min_results = []
#             all_max_results = []
#             all_std_results = []

#             for game_id, game in enumerate(desired_games):
#                 nr_perts = sorted(os.listdir(f'{path}/{algo}/{category}/{game}'))
#                 nr_perts = [0, 1, 2]

#                 for nr_pert in nr_perts:
#                     if not max_noises:
#                         max_noises = os.listdir(f'{path}/{algo}/{category}/{game}/{nr_pert}')

#                     if int(nr_pert) == nr_perts[0]:
#                         temp_len = 1 # len(desired_games)
#                         game_results = {noise: np.zeros((temp_len, )) for noise in max_noises if noise.startswith("0_")}
#                         min_results = {noise: np.array([1000000.0] * temp_len) for noise in max_noises if noise.startswith("0_")}
#                         max_results = {noise: np.array([-1000000.0] * temp_len) for noise in max_noises if noise.startswith("0_")}
#                         # max_noise_fully_filled = {noise: 0 for noise in max_noises if noise.startswith("0_")}
#                         all_results_for_plot = {noise: [[]] * temp_len for noise in max_noises if noise.startswith("0_")}
#                         std_results = {noise: np.zeros((temp_len, )) for noise in max_noises if noise.startswith("0_")}

#                     if len(max_noises) > 0:
#                         initial_results = np.array([0.0 for game in desired_games])
                        
#                         for max_noise in max_noises:
#                             test_case_path = f'{path}/{algo}/{category}/{game}/{nr_pert}/{max_noise}'
#                             desired_games_logs = [f'{game.lower()}.npy' for game in desired_games]
                            
#                             result = np.load(f"{test_case_path}/{game.lower()}.npy")
#                             trained_act_score = trained_dict[game]
#                             random_act_score = random_dict[game]

#                             if not isinstance(policy_id, list):
#                                 result = [[result[0][policy_id]]]
#                                 result = utils.normalize_result_for_one_policy(result, random_act_score[policy_id], trained_act_score[policy_id], game)
#                             else:
#                                 result = utils.normalize2(result, random_act_score, trained_act_score, game)

#                             game_id = 0
#                             game_results[max_noise][game_id] += result     
#                             min_results[max_noise][game_id] = min(min_results[max_noise][game_id], result)
#                             max_results[max_noise][game_id] = max(max_results[max_noise][game_id], result)
#                             all_results_for_plot[max_noise][game_id].append(result)
                
#                 nr_perts_len = len(nr_perts)
#                 for max_noise, v in game_results.items():
#                     game_results[max_noise] =  v / nr_perts_len
                
#                 for max_noise in max_noises:
#                     for g_id in range(1):
#                         std_results[max_noise][g_id] = np.std(all_results_for_plot[max_noise][g_id])
                
#                 if len(game_results.items()) > 0:
#                     all_games_results.append(game_results)
#                     all_min_results.append(min_results)
#                     all_max_results.append(max_results)
#                     all_std_results.append(std_results)

#     # return all_games_results, all_min_results, all_max_results, all_perts_games
#     return all_games_results, all_min_results, all_max_results, all_std_results, all_perts_games


# def get_results_for_random_games(algo, nr_perts_param=None, max_noises=None, log_games=None, policy_id=None):
#     category = "results"
#     path = "./final_results_0_policy/random"

#     all_baselines_trained_path = "./all_baselines_from_uber/trained"
#     all_baselines_random_path = "./all_baselines_from_uber/random"

#     if log_games:
#         desired_games = sorted(log_games)
#     else:
#         desired_games = sorted(utils.get_games_for_algo(algo))

#     for algo in [algo]:
#         all_perts_games = sorted(os.listdir(f'{path}/{algo}/{category}'))

#         trained_dict = {game: np.load(f"{all_baselines_trained_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
#         random_dict = {game: np.load(f"{all_baselines_random_path}/{game}_{algo}_policy_score.npy") for game in desired_games}
        
#         all_games_results = []
#         all_min_results = []
#         all_max_results = []
#         all_std_results = []

#         nr_perts = [0, 1, 2]
#         ultimate_game = "BankHeist"

#         for game_id, game in enumerate(desired_games):
#             for nr_pert in nr_perts:
#                 if not max_noises:
#                     max_noises = os.listdir(f'{path}/{algo}/{category}/{game}/{nr_pert}')

#                 if int(nr_pert) == nr_perts[0]:
#                     game_results = {noise: np.zeros((1, )) for noise in max_noises if noise.startswith("0_")}
#                     all_results_for_plot = {noise: [[]] * 1 for noise in max_noises if noise.startswith("0_")}
#                     std_results = {noise: np.zeros((1, )) for noise in max_noises if noise.startswith("0_")}

#                 if len(max_noises) > 0:
#                     for max_noise in max_noises:
#                         test_case_path = f'{path}/{algo}/{category}/{ultimate_game}/{nr_pert}/{max_noise}'
#                         desired_games_logs = [f'{game.lower()}.npy' for game in desired_games]
                            
#                         # print(f"{test_case_path}/{game.lower()}.npy")
#                         result = np.load(f"{test_case_path}/{game.lower()}.npy")
#                         trained_act_score = trained_dict[game]
#                         random_act_score = random_dict[game]

#                         if not isinstance(policy_id, list):
#                             result = [[result[0][policy_id]]]
#                             # print(random_act_score[policy_id], result, trained_act_score[policy_id])
#                             result = utils.normalize_result_for_one_policy(result, random_act_score[policy_id], trained_act_score[policy_id], game)
#                         else:
#                             result = utils.normalize2(result, random_act_score, trained_act_score, game)

#                         # print(result)
#                         # max_noise + nr_pert + game

#                         game_id = 0
#                         game_results[max_noise][game_id] += result   
#                         all_results_for_plot[max_noise][game_id].append(result)
                    
#         nr_perts_len = len(nr_perts)
#         for max_noise, v in game_results.items():
#             game_results[max_noise] =  v / nr_perts_len
                
#         for max_noise in max_noises:
#             for g_id in range(1):
#                 std_results[max_noise][g_id] = np.std(all_results_for_plot[max_noise][g_id])
                
#         if len(game_results.items()) > 0:
#             all_games_results.append(game_results)
#             all_std_results.append(std_results)

#     # print(all_std_results)
#     return all_games_results, [], [], all_std_results, []