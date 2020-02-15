import numpy as np
import sys
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import generate_rollout
from consts import *
from models import MakeAtariModel
import random_perturbation_generator as rpg

import tensorflow as tf

tf.compat.v1.enable_eager_execution()
# print(tf.test.gpu_device_name())

from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())


def run_experiments_for_env(game, algo, perturbation, nr_runs, eps, random_act):
    game = game[0].capitalize() + game[1:].lower()
    env = f"{game}NoFrameskip-v4"
    batch_size = 128
    run_ids = [0,1,2,3]
    tags = ["clear", "uni_pert", "final"]
    
    all_means, all_stddevs = [], []

    for policy_id in run_ids[1:]:
        results = []
        for run_id in range(nr_runs):
            print(f"Perturbation Test Algorithm: {algo} Environment: {env}")
            m = MakeAtariModel(algo,env,policy_id,tag="final")()
            results_final = generate_rollout.generate_rollout_with_eps_explorations(m, max_frames=2500, min_frames=2500,
                    perturbation=perturbation, random_act=random_act, epsilon_for_explorations=eps)
                
            results.append(results_final['ep_rewards'][0])
        
        results = np.array(results)
        all_means.append(np.mean(results))
        all_stddevs.append(np.std(results))
        print(f"Policy {policy_id} mean = {np.mean(results)}")
        print(f"Policy {policy_id} stddev = {np.std(results)}")

    return all_means, all_stddevs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("perturbation_file", type=str)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--game", type=str, default="seaquest")
    parser.add_argument("--random_act", type=bool, default=False)
    parser.add_argument("--nr_runs", type=int, default=1)
    parser.add_argument("--algo", type=str, default="rainbow")

    args = parser.parse_args()
    # perturbation = np.load(args.perturbation_file)
    means, stddevs = run_experiments_for_env(args.game, args.algo, None, args.nr_runs, args.eps, args.random_act)
    means = [str(m) for m in means]
    stddevs = [str(sd) for sd in stddevs]
    print("Means", " ".join(means))
    print("Stddevs", " ".join(stddevs))