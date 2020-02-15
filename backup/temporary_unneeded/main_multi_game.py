from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
import logging
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import three_games_training
import four_games_training
import two_games_training
import actions_and_obs_provider
import perturbation_test
import neptune_client

from dataset_loaders import *
from replay_buffer import *
from consts import *
from make_video import MakeVideo

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()

#TODO zrobić coś z games i envs z consts.py

algos = ['a2c','es','ga','apex','rainbow','dqn']
spaces = [ 
    np.linspace(0.0, 0.1, num=23),
    np.linspace(0.1, 0.15, num=4)[1:]
]
epsilons = np.concatenate(spaces)

lr = 0.001
batch_size = 128
run_ids = [0,1,2,3]
algo = 'rainbow'
tags = ["clear", "uni_pert", "final"]

max_noises = [0.001, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.05, 0.1, 0.5]

data_loader = TrajectoriesLoader(algo, True, epsilons)

for max_noise in max_noises:
    (observations, actions, game_labels), _ = data_loader.get_initial_buffer()
    nr_batches = actions.shape[0] // batch_size
    print("Number of random steps", actions.shape[0])
    print(f"Algorithm: {algo} Environment: many Noise max: {max_noise} lr: {lr}")

    models = [MakeAtariModel(algo,env,run_ids[2],tag="final")() for env in envs]
    results_noise = two_games_training.train_universal_perturbation_from_random_batches(models, max_frames=2500, min_frames=2500, 
            dataset=(observations, actions, game_labels), nr_batches=nr_batches, all_training_cases=data_loader.all_training_cases, 
            max_noise=max_noise, algo=algo, rep_buffer=data_loader.rep_buffer, lr=lr, data_loader=data_loader)
        
    perturbation = results_noise["perturbation"]

    results_to_log = perturbation_test.run_test_time_experiments(perturbation)
    print(f"{max_noise} training finished :)")
   
    enduro = perturbation_test.run_experiments_for_env(perturbation, "EnduroNoFrameskip-v4")
    pong = perturbation_test.run_experiments_for_env(perturbation, "PongNoFrameskip-v4")
    notseen = {"pong": pong, "enduro": enduro}

    neptune_client.neptune_backup_multi_game_training(batch_size, lr, results_to_log, results_noise, max_noise, notseen)
    
    for i in range(len(games)):
        make_video_for_game(envs[i], games[i], algo, noise_percent=max_noise, pert=perturbation)