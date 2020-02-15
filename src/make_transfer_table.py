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
import transfer_table_training as train_perturbation
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

# def get_random_pert(max_noise, random_path, nr_pert):
#     s_max_noise = str(max_noise).replace('.', '_')
#     if os.path.exists(f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy"):
#         print("Losowa perturbacja istnieje")
#         rand_pert = np.load(f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy")
#     else:
#         rand_pert = rpg.generate_random_perturbation(max_noise)
#         # print("HAHAHAHA", f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy")
#         np.save(f"{random_path}/pert_{s_max_noise}_{nr_pert}.npy", rand_pert)        

#     return rand_pert

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

def main(log_path, args, use_buffer=True):
    print(f"Log path: {log_path}")
    max_noises = [0.005, 0.008, 0.01, 0.05, 0.1]
    algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']
    lr = 0.1
    repeats = 1
    nr_test_runs = 1
    nr_different_perts_for_setup = 3
    batch_size = 128
    run_ids = [0,1,2,3]
    
    t_envs = [args.env]
    modes = [args.mode]
    algos = [args.algo]

    for algo in algos:
        print(f"Current algo: {algo}")
        envs = utils.get_sampled_games()
        random_policy_scores = pb.read_baselines_from_files("random", envs, algo)
        trained_policy_scores = pb.read_baselines_from_files("trained", envs, algo)

        for ii, mode in enumerate(modes):
            print(f"Current mode: {mode}")
            for game_id, game in enumerate(t_envs):
                env = f"{game[0].capitalize()}{game[1:]}NoFrameskip-v4"
                capitalized_game = game
                game = game.lower()
                print(f"Game {game} {env} now")

                for nr_pert in range(0, nr_different_perts_for_setup):
                    #if check_if_case_completed(f"{log_path}/{mode}/{algo}/results/{capitalized_game}/{nr_pert}"):
                    #   continue

                    for max_noise in max_noises:
                        s_m_n = str(max_noise).replace(".", "_")
                        utils.fix_path()
                        if max_noise == 0.0:
                            perturbation_test.run_experiments_for_env(None, env, algo=algo)
                            continue    
                    
                        current_pert = None
                        for nr_epoch in range(epochs): 
                            print(f"NR EPOCH {nr_epoch}")

                            if (mode == "trained") and nr_epoch == 0:
                                saved_trajectories = get_saved_dir(algo)
                                data_loader = TrajectoriesLoader([env], algo, False, args.policy_for_training, 
                                    saved_trajectories=saved_trajectories)

                            s_max_noise = str(max_noise).replace(".", "_")
                            normalized_results_path = f"{log_path}/{mode}/{algo}/normalized_results/{capitalized_game}/{nr_pert}/{s_max_noise}"
                            all_results_path = f"{log_path}/{mode}/{algo}/results/{capitalized_game}/{nr_pert}/{s_max_noise}"
                            pert_path = f"{log_path}/{mode}/{algo}/perts/{capitalized_game}/{nr_pert}/{s_max_noise}"

                            # clean_results_dirs_for_new_ones(normalized_results_path, all_results_path, pert_path)

                            utils.fix_path()
                            if not os.path.exists(pert_path):
                                if not os.path.exists(normalized_results_path):
                                    os.mkdir(normalized_results_path)
                                if not os.path.exists(all_results_path):
                                    os.mkdir(all_results_path)            
                                if mode == "trained" and not os.path.exists(pert_path):
                                    os.mkdir(pert_path)

                            if mode == "trained":
                                if os.path.exists(f"{pert_path}/pert.npy"): 
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
                                                data_loader=data_loader, n_repeats=repeats, seed=args.seed, pert_for_next_epoch=current_pert)
                                    else:
                                        loader = FullDatasetLoader([env], batch_size, algo) 
                                        results_noise = train_perturbation.train_universal_perturbation_from_full_dataset(m, 
                                                max_frames=2500, min_frames=2500, max_noise=max_noise, algo=algo, game=env, 
                                                loader=loader, n_repeats=repeats, seed=args.seed, pert_for_next_epoch=current_pert)
                                
                                    perturbation = results_noise["perturbation"] 
                                    open(f"{pert_path}/pert.npy", 'a').close()
                                    np.save(open(f"{pert_path}/pert.npy", "wb"), perturbation)
                            
                            elif mode == "random":
                                print(f"{log_path}/random/random_perts")
                                rand_pert = rpg.get_random_pert(max_noise, f"{log_path}/random/random_perts", nr_pert, args.seed)
                                perturbation = rand_pert
                        
                            print(f"We have perturbation for case {algo} {game} {nr_pert} {s_max_noise}")
                            
                            if mode == "trained" or mode == "random":
                                current_pert = perturbation
                                np.save(open(f"ten_perts/peturbation_{env}_{seed}_epoch_{nr_epoch}.npy", "wb"), current_pert)

                            for j, test_game in enumerate(envs):
                                print(f"Tests for game: {test_game}")
                        
                                test_env = f"{test_game[0].capitalize()}{test_game[1:]}NoFrameskip-v4" 
                                capitalized_test_game = game
                                test_game = test_game.lower()

                                all_results = []
                                mean_results = [0., 0., 0.]

                                for rep_id in range(nr_test_runs):
                                    results = perturbation_test.run_experiments_for_env(perturbation, test_env, algo=algo)
                                    results = [results[0][0], results[1][0], results[2][0]]
                                    all_results.append(results)
                                    
                                    for kk, res in enumerate(results):
                                        mean_results[kk] += res

                                print(f"{all_results_path}/{test_game}.npy")
                                np.save(open(f"{all_results_path}/{test_game}.npy", "wb"), np.array(all_results))

                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BankHeist")
    parser.add_argument("--algo", type=str, default="rainbow")
    parser.add_argument("--mode", type=str, default="trained")
    parser.add_argument("--policy_for_training", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(args.env, args.algo, args.mode, args.policy_for_training, args.seed)

    log_path = f"final_results_0_policy" #f"final_results"
    main(log_path, args)
