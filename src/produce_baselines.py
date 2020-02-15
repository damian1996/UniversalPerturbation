
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np

import perturbation_test
from consts import *
import random_perturbation_generator as rpg
import utils

import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.test.gpu_device_name()


def produce_baselines_from_uber(baseline_type, random_act, games=None):
    #all_games = games if games is not None else full_atari_games_list
    if baseline_type == "trained":
        repeats = 1
    else:
        repeats = 5

    for algo in algos:
        all_games = utils.get_games_for_algo(algo)

        for game in all_games:
            for nr in range(repeats):
                env = f"{game}NoFrameskip-v4"

                policy1, policy2, policy3 = [], [], []
                policy_scores = [policy1, policy2, policy3]

                results = perturbation_test.run_experiments_for_env(None, env, random_act=random_act, algo=algo)
                # print(results)
                policy_scores[0].append(results[0][0])
                policy_scores[1].append(results[1][0])
                policy_scores[2].append(results[2][0])

            mean1, mean2, mean3 = np.mean(np.array(policy_scores[0])), np.mean(np.array(policy_scores[1])), np.mean(np.array(policy_scores[2]))
            scores = np.array([mean1, mean2, mean3])
            #mean_score = int(np.mean(scores))
            #print(mean_score)
            utils.fix_path()
            np.save(open(f"all_baselines_from_uber/{baseline_type}/{game}_{algo}_policy_score.npy", "wb"), scores)

def read_baselines_from_files(baseline_type, games, algo):
    utils.fix_path()

    policy_scores = []
    for i, game in enumerate(games):
        mean_score = np.load(f"all_baselines_from_uber/{baseline_type}/{game}_{algo}_policy_score.npy") 
        #print("Mean score", mean_score)
        policy_scores.append(mean_score)

    return policy_scores

if __name__ == "__main__":
    #baseline_type = sys.argv[1]
    #random_act = sys.argv[2]
    #random_act = True if random_act == "true" else False 
    
    baseline_type = "trained"
    random_act = False
    produce_baselines_from_uber(baseline_type, random_act)
    
    baseline_type = "random"
    random_act = True
    produce_baselines_from_uber(baseline_type, random_act)

#     produce_baselines_from_uber("random", True)
#     produce_baselines_from_uber("trained", False)
   
#     trained_policy_scores = read_baselines_from_files("trained", full_atari_games_list)
#     random_policy_scores = read_baselines_from_files("random", full_atari_games_list)
