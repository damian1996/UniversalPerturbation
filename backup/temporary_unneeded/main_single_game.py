from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
import logging
import neptune

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
import single_game_training as train_perturbation
import actions_and_obs_provider

import perturbation_test
import neptune_client
import utils

from dataset_loaders import *

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()

game = "Pong" #sys.argv[1]
env = f"{game}NoFrameskip-v4"
# env = env[0].capitalize() + env[1:].lower()

spaces = [ 
    np.linspace(0.0, 0.1, num=35),
    np.linspace(0.1, 0.15, num=5)[1:]
]
epsilons = np.concatenate(spaces)

print(f"Game {env} now")
        
batch_size = 128
run_ids = [0,1,2,3]
algo = 'rainbow'
tags = ["clear", "uni_pert", "final"]
lr = 0.001
max_noises = [0.008]
# max_noises = [0.001, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.05, 0.1, 0.5]

# data_loader = TrajectoriesLoader(algo, False, epsilons)

for max_noise in max_noises: 
    rep_buffer = ReplayBuffer()

    all_training_cases = np.array([(env, eps) for eps in epsilons])
    np.random.shuffle(all_training_cases)
    observations, actions, _ = rep_buffer.generate_initial_dataset(all_training_cases, algo)

    # (observations, actions, _), _ = data_loader.get_initial_buffer()
    nr_batches = actions.shape[0] // batch_size
    print("Number of random steps", actions.shape[0])
        
    print(f"Algorithm: {algo} Environment: {env} Run Id: {run_ids[2]} Tag: final Noise max: {max_noise}")
    m = MakeAtariModel(algo,env,run_ids[2],tag="final")()

    results_noise = train_perturbation.train_universal_perturbation_from_random_batches(m, max_frames=2500, min_frames=2500, 
            dataset=(observations, actions, None), nr_batches=nr_batches, all_training_cases=all_training_cases, 
            max_noise=max_noise, algo=algo, rep_buffer=rep_buffer, game=env, data_loader=None)
    
    perturbation = results_noise["perturbation"]
    results_to_log = perturbation_test.run_experiments_for_env(perturbation, env)

    print(f"{max_noise} training finished :)")
    
    neptune_client.neptune_backup_single_game_training(batch_size, lr, results_to_log, results_noise, max_noise, "seaquest")

    make_video_for_game(env, game, algo, noise_percent=max_noise, pert=perturbation)