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
import train
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

def get_saved_dir(given_algo):
    algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']

    for ii, algo in enumerate(algos):
        if algo == given_algo:
            return f"./saved_trajectories_{ii}"

def dataset_size(s):
    smaller_eps, higher_eps = map(int, s.split(','))
    return smaller_eps, higher_eps


def main(log_path, args, use_buffer=True): 
    #print(f"Log path: {log_path}")
    utils.fix_path()
    
    # magnitudes = [0.005, 0.008, 0.01, 0.05, 0.1]
    # algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']
    run_ids = [0,1,2,3]
    # lr = 0.1
    
    algo = args.algo
    mode = args.mode
    game = args.env
    envs = [args.env] #utils.get_sampled_games()
    magnitude = args.magnitude

    random_policy_scores = pb.read_baselines_from_files("random", envs, algo)
    trained_policy_scores = pb.read_baselines_from_files("trained", envs, algo)
    
    env = f"{game[0].capitalize()}{game[1:]}NoFrameskip-v4"
    capitalized_game = game
    game = game.lower()
    
    nr_pert = 0
    s_magnitude = str(magnitude).replace(".", "_")
    # for nr_pert in range(0, args.nr_different_perts_for_setup):
        
    utils.fix_path()

    current_pert = None
    for nr_epoch in range(args.epochs): 
        print(f"NR EPOCH {nr_epoch}")

        if (mode == "trained") and nr_epoch == 0:
            saved_trajectories = get_saved_dir(algo)
            data_loader = TrajectoriesLoader(
                [env], algo, False, args.policy_for_training, args.seed, 
                args.batch_size, args.trajectories_at_once, args.nr_new_trajectories, 
                args.replay_after_batches, args.nr_of_all_trajectories, 
                saved_trajectories=saved_trajectories
            )

        if mode == "trained":
            print(f"Algorithm: {algo} Environment: {env} Run Id: {run_ids[2]} NrPert: {nr_pert} Noise max: {magnitude}")
            m = MakeAtariModel(algo,env,run_ids[2],tag="final")()
            
            if use_buffer:
                (observations, actions, _), _ = data_loader.get_initial_buffer()

                nr_batches = actions.shape[0] // args.batch_size
                #print("Number of random steps", actions.shape[0])

                utils.fix_path()
                                        
                results_noise = train.train_universal_perturbation_from_random_batches(m, 
                    max_frames=2500, min_frames=2500, dataset=(observations, actions, None), 
                    nr_batches=nr_batches, all_training_cases=data_loader.all_training_cases, 
                    max_noise=magnitude, algo=algo, rep_buffer=data_loader.rep_buffer, game=env, batch_size=args.batch_size,
                    data_loader=data_loader, n_repeats=args.repeats, seed=args.seed, pert_for_next_epoch=current_pert)
            else:
                loader = FullDatasetLoader([env], args.batch_size, algo) 
                results_noise = train.train_universal_perturbation_from_full_dataset(m, 
                        max_frames=2500, min_frames=2500, max_noise=magnitude, algo=algo, game=env, 
                        loader=loader, n_repeats=args.repeats, seed=args.seed, pert_for_next_epoch=current_pert)
            
            perturbation = results_noise["perturbation"] 
                                
            #open(f"{pert_path}/pert.npy", 'a').close()
            #np.save(open(f"{pert_path}/pert.npy", "wb"), perturbation)
                            
        elif mode == "random":
            rand_pert = rpg.get_random_pert(magnitude, f"{log_path}/random/random_perts", nr_pert, args.seed)
            perturbation = rand_pert     
            
        print(f"We have perturbation for case {algo} {game} {nr_pert} {s_magnitude}")
                            
        if mode == "trained" or mode == "random":
            current_pert = perturbation
            np.save(open(f"ten_perts/peturbation_{env}_{args.seed}_epoch_{nr_epoch}.npy", "wb"), current_pert)

        all_results = []
        for j, test_game in enumerate(envs):
            print(f"Tests for game: {test_game}")
                        
            test_env = f"{test_game[0].capitalize()}{test_game[1:]}NoFrameskip-v4" 
            capitalized_test_game = game
            test_game = test_game.lower()

            results = perturbation_test.run_experiments_for_env(perturbation, test_env, algo=algo)
            results = [results[0][0], results[1][0], results[2][0]]
            all_results.append(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BankHeist")
    parser.add_argument("--algo", type=str, default="rainbow")
    parser.add_argument("--mode", type=str, default="trained")
    parser.add_argument("--policy_for_training", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--magnitude", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nr_different_perts_for_setup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--trajectories_at_once", type=int, default=20) # 20
    parser.add_argument("--nr_new_trajectories", type=int, default=5) # 5
    parser.add_argument("--replay_after_batches", type=int, default=120) # 120
    parser.add_argument("--nr_of_all_trajectories", type=dataset_size, nargs=1, default=[(60, 9)]) # (60, 9)
    
    args = parser.parse_args()
    args.nr_of_all_trajectories = args.nr_of_all_trajectories[0]
    
    #print(args.env, args.algo, args.mode, args.policy_for_training, args.seed, args.magnitude)
    #print("Seen ", args.env, args.seed)
    print(f"Now lr: {args.lr}")
    log_path = f"final_results_0_policy" #f"final_results"
    main(log_path, args)
