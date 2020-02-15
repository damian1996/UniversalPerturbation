import generate_rollout
from consts import *
from models import MakeAtariModel

import numpy as np

def run_test_time_experiments(perturbation):
    batch_size = 128
    run_ids = [0,1,2,3]
    algo = 'rainbow'
    tags = ["clear", "uni_pert", "final"]
    all_results = []
    
    #TODO zaorac envs z consts
    for env in envs:
        results_to_log = run_experiments_for_env(perturbation, env)
        all_results.append(results_to_log)

    return all_results
        
def run_experiments_for_env(perturbation, env, random_act=False, algo=None, sticky_action_prob=0.0):
    batch_size = 128
    run_ids = [0,1,2,3]
    if algo is None:
        algo = 'rainbow'
    
    tags = ["clear", "uni_pert", "final"]

    results_to_log = []
        
    final_run_id = run_ids[1]
    print(f"Perturbation Test Algorithm: {algo} Environment: {env}")
    m = MakeAtariModel(algo,env,final_run_id,tag="final")()
    results_final = generate_rollout.generate_clean_rollout(m, max_frames=2500, min_frames=2500,
            perturbation=perturbation, random_act=random_act, sticky_action_prob=sticky_action_prob)
    print("Not seen run 1", results_final['ep_rewards'])
    results_to_log.append(results_final['ep_rewards'])
    
    final_run_id = run_ids[2]
    m = MakeAtariModel(algo,env,final_run_id,tag="final")()
    results_final = generate_rollout.generate_clean_rollout(m, max_frames=2500, min_frames=2500,
            perturbation=perturbation, random_act=random_act, sticky_action_prob=sticky_action_prob)
    print("Seen run 2", results_final['ep_rewards'])
    results_to_log.append(results_final['ep_rewards'])

    final_run_id = run_ids[3]
    m = MakeAtariModel(algo,env,final_run_id,tag="final")()
    results_final = generate_rollout.generate_clean_rollout(m, max_frames=2500, min_frames=2500,
            perturbation=perturbation, random_act=random_act, sticky_action_prob=sticky_action_prob)
    print("Seen run 3", results_final['ep_rewards'])
    results_to_log.append(results_final['ep_rewards'])

    return results_to_log