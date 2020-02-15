from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
from easydict import EasyDict

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
import train as train_perturbation

import neptune_client
import random_perturbation_generator as rpg
import utils

from dataset_loaders import *
from consts import *

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()


def get_random_pert(max_noise, random_path, nr_pert):
    s_max_noise = str(max_noise).replace('.', '_')
    return rpg.generate_random_perturbation(max_noise)

def create_all_random_perts(max_noises, nr_different_perts_for_setup, log_path):
    utils.fix_path()
    for max_noise in max_noises:
        for nr_pert in range(nr_different_perts_for_setup):
            print(nr_pert, max_noise)
            rand_pert = get_random_pert(max_noise, f"{log_path}/random/random_perts", nr_pert) 

def get_saved_dir(given_algo):
    algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']

    for ii, algo in enumerate(algos):
        if algo == given_algo:
            return f"./saved_trajectories_{ii}"

def compute_one_perturbation(algo, mode, nr_pert, max_noise, env, use_buffer=True):
    m = MakeAtariModel(algo,env,run_ids[2],tag="final")()

    if use_buffer:
        saved_trajectories = get_saved_dir(algo)
        data_loader = TrajectoriesLoader([env], algo, False, saved_trajectories=saved_trajectories)
        
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
    return perturbation

def save_perturbation(algo, mode, nr_pert, max_noise, env, hyperparams):
    if mode == "trained":
        print(f"Algorithm: {algo} Environment: {env} Run Id: {run_ids[2]} NrPert: {nr_pert} Noise max: {max_noise}")
        perturbation = compute_one_perturbation(algo, mode, nr_pert, max_noise, env)
    elif mode == "random":
        print(f"{log_path}/random/random_perts")
        perturbation = get_random_pert(max_noise, f"{log_path}/random/random_perts", nr_pert)

    perturbation = perturbation.reshape(1, 84, 84, 1)

    if os.path.exists(f'placeholder/pert_{algo}_{mode}_{nr_pert}_{max_noise}.npy'):
        recent_perts = np.load(f'placeholder/pert_{algo}_{mode}_{nr_pert}_{max_noise}.npy')
    
    if recent_perts:
        all_perts = np.concatenate((recent_perts, perturbation), axis=0)
    else:
        all_perts = perturbation

    np.save(f'placeholder/pert_{algo}_{mode}_{nr_pert}_{max_noise}.npy', all_perts)


if __name__ == '__main__':
    hyperparams = EasyDict({
        'lr': 0.1,
        'repeats': 1,
        'nr_test_runs': 1,
        'nr_different_perts_for_setup': 3,
        'batch_size': 128,
        'run_ids': [0,1,2,3]
    })

    algo = sys.argv[1]
    mode = sys.argv[2]
    nr_pert = sys.argv[3]
    max_noise = sys.argv[4]
    env = sys.argv[5]
    print(algo, mode, nr_pert, max_noise, env)

    save_perturbation(algo, mode, nr_pert, max_noise, env, hyperparams)