from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
import logging
import argparse
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("..")
os.chdir("atari-model-zoo")

import atari_zoo
import atari_zoo.config
from atari_zoo.utils import *
import atari_zoo.atari_wrappers as atari_wrappers
from atari_zoo.dopamine_preprocessing import AtariPreprocessing as DopamineAtariPreprocessing	
from atari_zoo.atari_wrappers import FireResetEnv, NoopResetEnv, MaxAndSkipEnv,WarpFrameTF,FrameStack,ScaledFloatFrame

from models import MakeAtariModel
import generate_rollout
import train as train_perturbation
import actions_and_obs_provider

import perturbation_test
import neptune_client
import random_perturbation_generator as rpg
import utils

from dataset_loaders import *
from consts import *
import produce_baselines as pb

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()

def get_random_pert(max_noise, random_path, nr_pert):
    s_max_noise = str(max_noise).replace('.', '_')
    if os.path.exists(f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy"):
        print("Losowa perturbacja istnieje")
        rand_pert = np.load(f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy")
    else:
        rand_pert = rpg.generate_random_perturbation(max_noise)
        # print("HAHAHAHA", f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy")
        np.save(f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy", rand_pert)        

    return rand_pert

def get_saved_dir(given_algo):
    algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']

    for ii, algo in enumerate(algos):
        if algo == given_algo:
            return f"./saved_trajectories_{ii}"

def create_all_random_perts(max_noises, nr_different_perts_for_setup, log_path):
    utils.fix_path()
    for max_noise in max_noises:
        for nr_pert in range(nr_different_perts_for_setup):
            print(nr_pert, max_noise)
            rand_pert = get_random_pert(max_noise, f"{log_path}/random/random_perts", nr_pert) 
    
def check_if_case_completed(path, all_exps_len=20):
    max_noises = os.listdir(path)
    print("BRRRRRRR") 
    if len(max_noises) == 1 or (len(max_noises) == 2 and "0_008" in max_noises):
        return False
    
    for max_noise in max_noises:
        if ".py" in max_noise:
            continue         
        noise_path = f"{path}/{max_noise}"
        print(noise_path)

        if not os.path.exists(noise_path):
            return False

        experiments = os.listdir(noise_path)

        if (len(experiments) < all_exps_len) or (len(experiments) > all_exps_len):
            print("BRRRR ", path)
            return False
    
    return True

def clean_results_dirs_for_new_ones(normalized_results_path, all_results_path, pert_path, all_exps_len=20):
    if os.path.exists(normalized_results_path):
        results = os.listdir(normalized_results_path)
        print("Normalized results len ", len(results))
        if len(results) < all_exps_len:
            shutil.rmtree(normalized_results_path)
    
    if os.path.exists(all_results_path):
        results = os.listdir(all_results_path)
        print("Results len ", len(results))
        if len(results) < all_exps_len:
            shutil.rmtree(all_results_path)
    
    if os.path.exists(pert_path):
        shutil.rmtree(pert_path)

def main(log_path, train_env, algo, mode, special_label_for_neptun, use_buffer=True):
    print(f"Log path: {log_path}")
    max_noises = [0.005, 0.008, 0.01, 0.05, 0.1]
    algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']
    lr = 0.1
    repeats = 1
    nr_test_runs = 1
    nr_different_perts_for_setup = 3
    batch_size = 128
    run_ids = [0,1,2,3]
    
    modes = [mode]
    algos = [algo]
    test_envs = utils.get_sampled_games()

    for algo in algos:
        print(f"Current algo: {algo}")
        random_policy_scores = pb.read_baselines_from_files("random", test_envs, algo)
        trained_policy_scores = pb.read_baselines_from_files("trained", test_envs, algo)

        for ii, mode in enumerate(modes):
            print(f"Current mode: {mode}")
            
            for nr_pert in range(0, nr_different_perts_for_setup):        
                s_m_n = str(max_noises[0]).replace(".", "_")
                utils.fix_path()
                    
                for max_noise in max_noises:
                    env = f"{train_env[0].capitalize()}{train_env[1:]}NoFrameskip-v4"
                    capitalized_game = train_env

                    if max_noise == 0.0:
                        perturbation_test.run_experiments_for_env(None, env, algo=algo)
                        continue
                    elif check_if_case_completed(algo, mode, nr_pert, max_noise, special_label_for_neptun):
                        continue

                    if check_if_case_started(algo, mode, nr_pert, max_noise):
                        case_reloaded = True
                        new_state, old_state = load_state()
                        (algo, mode, nr_pert, max_noise) = new_state

                    if (mode == "trained"):
                        saved_trajectories = get_saved_dir(algo)
                        data_loader = TrajectoriesLoader([env], algo, False, saved_trajectories=saved_trajectories)

                    s_max_noise = str(max_noise).replace(".", "_")
                    utils.fix_path()

                    # load perturbation - wywalic do funkcji
                    if mode == "trained":
                        if if_pert_exists(...): # os.path.exists(f"{pert_path}/pert.npy"): 
                            print("Pert already exists")
                            perturbation = np.load(f"{pert_path}/pert.npy")
                        else:
                            print(f"Algorithm: {algo} Environment: {env} Run Id: {run_ids[2]} NrPert: {nr_pert} Noise max: {max_noise}")
                            m = MakeAtariModel(algo,env,run_ids[2],tag="final")()
						
                            if use_buffer:
                                (observations, actions, _), _ = data_loader.get_initial_buffer()

                                nr_batches = actions.shape[0] // batch_size
                                print("Number of random steps", actions.shape[0])

                                utils.fix_path()
                                    
                                results_noise = train_perturbation.train_universal_perturbation_from_random_batches(m, 
                                    max_frames=2500, min_frames=2500, dataset=(observations, actions, None), 
                                    nr_batches=nr_batches, all_training_cases=data_loader.all_training_cases, 
                                    max_noise=max_noise, algo=algo, rep_buffer=data_loader.rep_buffer, game=env, 
                                    data_loader=data_loader, n_repeats=repeats)
                            else:
                                loader = FullDatasetLoader([env], batch_size, algo) 
                                results_noise = train_perturbation.train_universal_perturbation_from_full_dataset(m, 
                                    max_frames=2500, min_frames=2500, max_noise=max_noise, algo=algo, game=env, 
                                    loader=loader, n_repeats=repeats)
							
                            perturbation = results_noise["perturbation"]
                            # open(f"{pert_path}/pert.npy", 'a').close()
                            # np.save(open(f"{pert_path}/pert.npy", "wb"), perturbation)
                        
                    elif mode == "random":
                        print(f"{log_path}/random/random_perts")
                        rand_pert = get_random_pert(max_noise, f"{log_path}/random/random_perts", nr_pert)
                        perturbation = rand_pert
					
                    perts.append(perturbation)
                    np.save(open(f"placeholder/pert_{mode}_{algo}_{max_noise}_{nr_pert}.npy", "wb"), np.array(perts))

                    print(f"We have perturbation for case {algo} {game} {nr_pert} {s_max_noise}")

                    results_for_perturbation = []
                    for j, test_game in enumerate(test_envs):
                        print(f"Tests for game: {test_game}")
					
                        test_env = f"{test_game[0].capitalize()}{test_game[1:]}NoFrameskip-v4" 
                        capitalized_test_game = game

                        all_results, mean_results = [], [0., 0., 0.]

                        for rep_id in range(nr_test_runs):
                            results = perturbation_test.run_experiments_for_env(perturbation, test_env, algo=algo)
                            results = [results[0][0], results[1][0], results[2][0]]
                            all_results.append(results)
                                
                            for kk, res in enumerate(results):
                                mean_results[kk] += res

                        results_for_perturbation.append(np.array(all_results))
                        
                    transfer_results.append(np.array(results_for_perturbation))

                    np.save(
                        open(f"placeholder/transfer_table_{mode}_{algo}_{max_noise}_{nr_pert}.npy", "wb"), 
                        np.array(transfer_results)
                    )

                transfer_results = np.array(transfer_results)
                perts = np.array(perts)
                neptune_client.save_transfer_data(transfer_results, perts, mode, algo, nr_pert, max_noise, games, special_label_for_neptun)
                    
                if case_reloaded:
                        # przywroc stare parametry algo, mode, nr_pert i magnitude
                        print("LALALA")

                clean_placeholders(mode, algo, max_noise, nr_pert)
                        
if __name__ == '__main__':
    special_label_for_neptun = "special_label_for_neptun"
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BankHeist")
    parser.add_argument("--algo", type=str, default="rainbow")
    parser.add_argument("--mode", type=str, default="trained")
    args = parser.parse_args()
    print(args.env, args.algo, args.mode)

    log_path = f"final_results_0_policy" #f"final_results"
    main(log_path, args.env, args.algo, args.mode, special_label_for_neptun)
