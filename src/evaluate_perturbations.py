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

def main(log_path, algo, mode, special_label_for_neptun, use_buffer=True):
    print(f"Log path: {log_path}")
    max_noises = [0.005, 0.008, 0.01, 0.05, 0.1]
    hyperparams = EasyDict({
        'lr': 0.1,
        'repeats': 1,
        'nr_test_runs': 1,
        'nr_different_perts_for_setup': 3,
        'batch_size': 128,
        'run_ids': [0,1,2,3]
    })

    modes = [mode]
    algos = [algo]
    envs = utils.get_sampled_games()

    for algo in algos:
        print(f"Current algo: {algo}")
        random_policy_scores = pb.read_baselines_from_files("random", envs, algo)
        trained_policy_scores = pb.read_baselines_from_files("trained", envs, algo)

        for ii, mode in enumerate(modes):
            print(f"Current mode: {mode}")
            
            for nr_pert in range(0, hyperparams.nr_different_perts_for_setup):        
                s_m_n = str(max_noises[0]).replace(".", "_")
                utils.fix_path()
                    
                for max_noise in max_noises:
                    perturbations = neptune_client.get_perturbations(algo, mode, nr_pert, max_noise, special_label_for_neptun)
                    transfer_results = []

                    for game_id, game in enumerate(envs):
                        env = f"{game[0].capitalize()}{game[1:]}NoFrameskip-v4"

                        if max_noise == 0.0:
                            perturbation_test.run_experiments_for_env(None, env, algo=algo)
                            continue

                        s_max_noise = str(max_noise).replace(".", "_")
                        utils.fix_path()

                        results_for_perturbation = []
                        for test_game_id, test_game in enumerate(envs):
                            print(f"Tests for game: {test_game}")
                            test_env = f"{test_game[0].capitalize()}{test_game[1:]}NoFrameskip-v4" 

                            all_results, mean_results = [], [0., 0., 0.]
                            for rep_id in range(hyperparams.nr_test_runs):
                                results = perturbation_test.run_experiments_for_env(perturbation, test_env, algo=algo)
                                results = [results[0][0], results[1][0], results[2][0]]
                                all_results.append(results)
                                    
                                for kk, res in enumerate(results):
                                    mean_results[kk] += res

                            results_for_perturbation.append(np.array(all_results))
                            
                        transfer_results.append(np.array(results_for_perturbation))

                    transfer_results = np.array(transfer_results)
                    neptune_client.save_transfer_data(transfer_results, mode, algo, nr_pert, max_noise, envs, special_label_for_neptun)
                        
                        
if __name__ == '__main__':
    special_label_for_neptun = "special_label_for_neptun"
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="rainbow")
    parser.add_argument("--mode", type=str, default="trained")
    args = parser.parse_args()
    print(args.algo, args.mode)

    log_path = f"final_results_0_policy" #f"final_results"
    main(log_path, args.algo, args.mode, special_label_for_neptun)