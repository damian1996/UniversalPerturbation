def get_results_with_old_baselines(path, algo):
    categories = ["results", "normalized_results", "perts"]

    for algo in [algo]:
        for category in categories:
            if category != "normalized_results":
                continue
            
            all_games_results = []
            all_perts_games = os.listdir(f'{path}/{algo}/{category}')
            
            for game in all_perts_games:
                nr_perts = sorted(os.listdir(f'{path}/{algo}/{category}/{game}'))
                for nr_pert in nr_perts:
                    max_noises = os.listdir(f'{path}/{algo}/{category}/{game}/{nr_pert}')
                    if int(nr_pert) == 0:
                        game_results = {noise: np.zeros((len(all_perts_games), )) for noise in max_noises if noise.startswith("0_")}
                        print(game_results)
                        max_noise_fully_filled = {noise: 0 for noise in max_noises if noise.startswith("0_")}
                        print(game_results.items())

                    if len(max_noises) > 1:                        
                        for max_noise in max_noises:
                            if ".py" in max_noise:
                                continue
                            
                            test_case_path = f'{path}/{algo}/{category}/{game}/{nr_pert}/{max_noise}'
                            result_games = os.listdir(test_case_path)
                            for g_id, res_game in enumerate(result_games):
                                result = np.load(f"{test_case_path}/{res_game}")
                                game_results[max_noise][g_id] += result

                        max_noise_fully_filled[max_noise] += 1            
                        
                
                for max_noise, v in game_results.items():
                    game_results[max_noise] =  v / max_noise_fully_filled[max_noise]
                
                all_games_results.append(game_results)
                        
    return all_games_results, all_perts_games

def get_normalized_result_old_baselines(path, algo):
    results, _ = get_results_with_old_baselines(path, algo)
    print(np.mean(np.array(results)))
    return np.mean(np.array(results))

def get_transfer_table_old_baselines(path, algo):
    results, games = get_results_with_old_baselines(path, algo)
    print(np.array(results).shape)
    return np.array(results)
