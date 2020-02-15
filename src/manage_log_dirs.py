import os
import shutil

import utils
from consts import *

def make_logs_dir():
    nr_perts = 10
    logs_dir = "final_results/trained"
    os.mkdir(logs_dir)

    paths = ["rainbow", "dqn", "ga", "a2c", "impala", "es", "apex"]
    paths = [f"./{logs_dir}/{path}" for path in paths]
    for path in paths:
        os.mkdir(path)
    print(paths)

    results_dirs = ["normalized_results", "results", "perts"]

    for log_path in paths:
        for result_dir in results_dirs:
            case_dir = f"{log_path}/{result_dir}"
            print(case_dir)
            os.mkdir(case_dir)
            for game in full_dopamine_games_list:
                game_dir = f"{case_dir}/{game}"
                print(game_dir)
                os.mkdir(game_dir)
                print(f"Created {game_dir}")

                for nr in range(nr_perts):
                    pert_dir = f"{game_dir}/{str(nr)}"
                    os.mkdir(pert_dir)
                    open(f"{pert_dir}/empty.py", 'a').close()

    '''
    nr_perts = 10
    logs_dir = "final_results/random"
    paths = os.listdir(f"./{logs_dir}")
    paths = [f"./{logs_dir}/{path}" for path in paths]
    print(paths)

    results_dirs = ["normalized_results", "results"]

    for log_path in paths:
        for result_dir in results_dirs:
            case_dir = f"{log_path}/{result_dir}"
            
            for game in full_dopamine_games_list:
                game_dir = f"{case_dir}/{game}"
                os.mkdir(game_dir)
                print(f"Created {game_dir}")

                for nr in range(nr_perts):
                    pert_dir = f"{game_dir}/{str(nr)}"
                    os.mkdir(pert_dir)
                    open(f"{pert_dir}/empty.py", 'a').close()
    '''

def delete_unneeded_logs(main_dir, algos, modes, pert_nrs):
    for mode in modes:
        for algo in algos:
            logs_dir = f"./{main_dir}/{mode}/{algo}"
            log_types = os.listdir(logs_dir)
            
            for log_type in log_types:
                log_type_dir = f"{logs_dir}/{log_type}"
                games = os.listdir(log_type_dir)

                for game in games:
                    game_dir = f"{log_type_dir}/{game}"
                    
                    for pert_nr in pert_nrs:
                        pert_logs_dir = f"{game_dir}/{pert_nr}"
                        game_logs = os.listdir(pert_logs_dir)
                        if len(game_logs) > 1:
                            for game_log_dir in game_logs:
                                if game_log_dir.startswith("0_"):
                                    if "/perts" not in f"{pert_logs_dir}":
                                        print(f"{pert_logs_dir}/{game_log_dir}")
                                        shutil.rmtree(f"{pert_logs_dir}/{game_log_dir}")
                                    # os.rmdir(f"{pert_logs_dir}/{game_log_dir}")


def get_logs(main_dir, algos, modes, pert_nrs):
    for mode in modes:
        for algo in algos:
            all_envs = utils.get_games_for_algo(algo)
            logs_dir = f"./{main_dir}/{mode}/{algo}"
            log_types = os.listdir(logs_dir)

            for log_type in log_types:
                log_type_dir = f"{logs_dir}/{log_type}"
                games = os.listdir(log_type_dir)

                for game in games:
                    if game not in all_envs:
                        continue

                    game_dir = f"{log_type_dir}/{game}"

                    for pert_nr in pert_nrs:
                        pert_logs_dir = f"{game_dir}/{pert_nr}"
                        #print(pert_logs_dir)
                        game_logs = os.listdir(pert_logs_dir)
                        if len(game_logs) > 1:
                            for game_log_dir in game_logs:
                                if game_log_dir.startswith("0_"):
                                    length = len(os.listdir(f"{pert_logs_dir}/{game_log_dir}"))
                                
                                if game_log_dir.startswith("0_") and length < 20 and "perts" not in pert_logs_dir:
                                    print(f"{pert_logs_dir}/{game_log_dir} {length}")
                        else:
                            print(pert_logs_dir)


if __name__ == '__main__': 
    # delete_unneeded_logs("final_results_without_buffer", ["dqn"], ["random"], [0, 1, 2, 3, 4])
    delete_unneeded_logs("final_results", ["a2c"], ["trained"], [0, 1, 2, 3, 4])

    # get_logs("final_results_without_buffer", ["rainbow"], ["trained"], [0, 1])
    # get_logs("final_results", ["rainbow"], ["trained"], [0, 1, 2, 3, 4])
    # get_logs("final_results", ["dqn"], ["trained"], [0, 1, 2, 3, 4])
