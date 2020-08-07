import numpy as np
import sys
import argparse
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import generate_rollout
from consts import *
from models import MakeAtariModel
import produce_baselines as pb
import random_perturbation_generator as rpg
import utils

tf.compat.v1.enable_eager_execution()
# print(tf.test.gpu_device_name())

from tensorflow.python.client import device_lib 


# policy_id jest w przedziale [1,3], a nie od 0 :P
def eval_perturbation(game_name, algo, policy_id, perturbation=None, normalize_or_not=False, random_act=False):
    game_name = game_name[0].capitalize() + game_name[1:].lower()
    env = f"{game_name}NoFrameskip-v4"
    batch_size = 128
    run_ids = [0,1,2,3]
    tags = ["clear", "uni_pert", "final"]
    
    results = []
    print(f"Perturbation Test Algorithm: {algo} Environment: {env} Policy: {policy_id}")
    m = MakeAtariModel(algo,env,policy_id,tag="final")()
    results_final = generate_rollout.generate_rollout_with_eps_explorations(m, max_frames=2500, min_frames=2500,
            perturbation=perturbation, random_act=random_act)
    
    if normalize_or_not:
        random_policy_score = pb.read_baselines_from_files("random", [game_name], algo)[0][policy_id-1]
        trained_policy_score = pb.read_baselines_from_files("trained", [game_name], algo)[0][policy_id-1]
        return utils.normalize_results(results_final['ep_rewards'], [random_policy_score], [trained_policy_score])
    else:
        return results_final['ep_rewards'][0]

def eval_perturbation_for_all_policies(game_name, algo, perturbation=None, normalize_or_not=False, random_act=False):
    game_name = game_name[0].capitalize() + game_name[1:].lower()
    env = f"{game_name}NoFrameskip-v4"
    batch_size = 128
    run_ids = [0,1,2,3]
    tags = ["clear", "uni_pert", "final"]
    
    results = []
    for policy_id in range(1, 4):
        print(f"Perturbation Test Algorithm: {algo} Environment: {env} Policy: {policy_id}")
        m = MakeAtariModel(algo,env,policy_id,tag="final")()
        results_final = generate_rollout.generate_rollout_with_eps_explorations(m, max_frames=2500, min_frames=2500,
                perturbation=perturbation, random_act=random_act)
        
        results.append(results_final['ep_rewards'][0])

    if normalize_or_not:
        random_policy_scores = pb.read_baselines_from_files("random", [game_name], algo)[0]
        trained_policy_scores = pb.read_baselines_from_files("trained", [game_name], algo)[0]
        return utils.normalize_results(results, random_policy_scores, trained_policy_scores)
    else:
        return results_final['ep_rewards'][0]

if __name__ == "__main__":
    game_name = "krull"
    algo = "rainbow"
    policy_id = 1
    random_act = False
    normalize_or_not=True
    nr_runs = 1
    perturbation = None

    results = eval_perturbation(
        game_name, 
        algo, 
        policy_id, 
        perturbation=perturbation,
        normalize_or_not=normalize_or_not,
        random_act=random_act
    )

    results = eval_perturbation_for_all_policies(
        game_name,
        algo,
        perturbation=perturbation,
        normalize_or_not=normalize_or_not,
        random_act=random_act
    )