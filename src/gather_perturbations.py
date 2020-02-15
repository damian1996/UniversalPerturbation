from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np
import argparse

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

import neptune_client
import utils

from dataset_loaders import *
from consts import *

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()

def clean_placeholders(perts_path):
    os.remove(perts_path)

def remove_already_computed_envs(perts_path, envs):
    perts = np.load(perts_path)
    nr_computed_perts = perts.shape[0]
    return envs, envs[nr_computed_perts:]

def check_if_case_started(perts_path):
    return os.path.exists(perts_path)

def collect_perturbations_for_all_cases(algo, mode, special_label_for_neptun, use_buffer=True):
    print(f"Log path: {log_path}")
    nr_different_perts_for_setup = 3
    
    modes = [mode]
    algos = [algo]
    max_noises = [0.005, 0.008, 0.01, 0.05, 0.1]
    envs = utils.get_sampled_games()

    for algo in algos:
        print(f"Current algo: {algo}")
        for ii, mode in enumerate(modes):
            print(f"Current mode: {mode}")
            
            for nr_pert in range(0, nr_different_perts_for_setup):        
                s_m_n = str(max_noises[0]).replace(".", "_")
                utils.fix_path()
                    
                for max_noise in max_noises:
                    perts_path = f'placeholder/pert_{algo}_{mode}_{nr_pert}_{max_noise}.npy'

                    if is_perturbation_computed(algo, mode, nr_pert, max_noise, special_label_for_neptun):
                        continue

                    if check_if_case_started(perts_path):
                        is_reloaded = True
                        old_envs, envs = remove_already_computed_envs(perts_path, envs)                        

                    for game_id, game in envs:
                        env = f"{game[0].capitalize()}{game[1:]}NoFrameskip-v4"
                        capitalized_game = game

                        s_max_noise = str(max_noise).replace(".", "_")
                        utils.fix_path()

                        subprocess.call(['python3', 'compute_perturbation.py', algo, mode, nr_pert, max_noise, env])
                        
                        # np.save(open(f"placeholder/pert_{mode}_{algo}_{max_noise}_{nr_pert}.npy", "wb"), np.array(perts))
                        print(f"We have perturbation for case {algo} {game} {nr_pert} {s_max_noise}")

                    perts = np.load(perts_path)
                    neptune_client.save_perturbation(perts, mode, algo, nr_pert, max_noise, special_label_for_neptun)

                    if is_reloaded:
                        is_reloaded = False
                        envs = old_envs
                        
                    clean_placeholder(perts_path)
            
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="BankHeist")
    parser.add_argument("--algo", type=str, default="rainbow")
    parser.add_argument("--mode", type=str, default="trained")
    parser.add_argument("--label", type=str, default="special_label")
    args = parser.parse_args()
    print(args.algo, args.mode, args.label)

    log_path = f"final_results_0_policy" #f"final_results"
    collect_perturbations_for_all_cases(args.algo, args.mode, args.label)