import numpy as np
import tensorflow as tf

import atari_zoo
from models import MakeAtariModel

import generate_rollout

def get_all_training_actions_and_observations(env, algo, best_id=None):
    run_ids = [0,1,2,3]
    tags = ["clear", "uni_pert", "final"]
    all_trajectories = {'observations': [], 'actions': []}
    chosen_run_ids = run_ids[2:]
    if best_id:
        chosen_run_ids = [best_id]

    for run_id in chosen_run_ids:
        print('Algorithm: {} Environment: {} Run Id: {} Tag: {}'.format(algo,env,run_id,"final"))
        m = MakeAtariModel(algo,env,run_id,tag="final")()
        results_clean = generate_rollout.generate_clean_rollout(m, max_frames=2500, min_frames=2500)
        
        act = results_clean['actions']
        obs = results_clean['observations']

        for i in range(len(act)):
            all_trajectories['observations'].append(obs[i])
            all_trajectories['actions'].append(act[i])

    actions = all_trajectories['actions']
    observations = all_trajectories['observations']
    indexes = np.arange(0, len(actions))
    np.random.shuffle(indexes)
    actions2 = np.array(actions)[indexes]
    observations = np.array(observations)[indexes]

    return observations, actions2, actions


def get_all_actions_and_observations_with_epsilon_explorations(env, algo, eps, best_id=None):
    run_ids = [0,1,2,3]
    tags = ["clear", "uni_pert", "final"]
    all_trajectories = {'observations': [], 'actions': []}
    
    chosen_run_ids = run_ids[2:]
    if best_id:
        chosen_run_ids = [best_id]

    for run_id in chosen_run_ids:
        print('Algorithm: {} Environment: {} Run Id: {} Tag: {} Eps: {}'.format(algo,env,run_id,"final",eps))
        m = MakeAtariModel(algo,env,run_id,tag="final")()
        results_clean = generate_rollout.generate_rollout_with_eps_explorations(m, max_frames=2500, min_frames=2500, epsilon_for_explorations=eps)
        
        act = results_clean['actions']
        obs = results_clean['observations']

        for i in range(len(act)):
            all_trajectories['observations'].append(obs[i])
            all_trajectories['actions'].append(act[i])

    actions = all_trajectories['actions']
    observations = all_trajectories['observations']
    indexes = np.arange(0, len(actions))
    np.random.shuffle(indexes)
    actions2 = np.array(actions)[indexes]
    observations = np.array(observations)[indexes]

    print("Shapes", observations.shape, actions2.shape)

    return observations, actions2, actions