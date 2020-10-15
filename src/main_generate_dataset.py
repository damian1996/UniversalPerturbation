from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
import logging
import argparse
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import transfer_table_training
import actions_and_obs_provider

import perturbation_test
import neptune_client
import random_perturbation_generator as rpg
import utils

from generate_better_dataset import *
from consts import *
import produce_baselines as pb

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()

def get_saved_dir(given_algo):
    algos = ['rainbow', 'dqn', 'ga', 'es', 'impala', 'a2c']

    for ii, algo in enumerate(algos):
        if algo == given_algo:
            return f"./saved_trajectories_{ii}"

def main(args, use_buffer=True): 
    utils.fix_path()

    run_ids = [0,1,2,3]
    algos = [args.algo]

    for algo in algos:
        print(f"Current algo: {algo}")
        env = f"{args.env}NoFrameskip-v4"

        saved_trajectories = get_saved_dir(algo)
        data_loader = MultiDatasetLoader([env], algo, False, args.policy_for_training, args.seed, 
            saved_trajectories=saved_trajectories)

        m = MakeAtariModel(algo,env,run_ids[2],tag="final")()
                                
        utils.fix_path()
                                        
        transfer_table_training.generate_more_informative_dataset(m, 
            max_frames=2500, min_frames=2500, algo=algo, game=args.env, 
            data_loader=data_loader, seed=args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BankHeist")
    parser.add_argument("--algo", type=str, default="rainbow")
    parser.add_argument("--mode", type=str, default="trained")
    parser.add_argument("--policy_for_training", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(args.env, args.algo, args.mode, args.policy_for_training, args.seed)

    print("Seen ", args.env, args.seed)
    main(args)
