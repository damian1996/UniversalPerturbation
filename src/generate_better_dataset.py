import numpy as np
import tensorflow as tf
from numpy.random import default_rng
import os

from consts import *
import utils
import actions_and_obs_provider

class ChunksCreator:

    def __init__(self, envs, nr_policy, seed, multi_game=False):
        self.rng = default_rng(seed)
        self.multi_game = multi_game
        self.d_games = {env: i for i, env in enumerate(envs)}

        self.policy_id = 2
        self.batch_counter = 0

    def generate_trajectories_for_given_cases(self, data_cases, algo):      
        l_actions, l_observations, l_game_labels = [], [], []

        for i, (env, eps) in enumerate(data_cases): 
            best_policy_id = self.policy_id 
            observations_one_game, actions_one_game, _ = actions_and_obs_provider.get_all_actions_and_observations_with_epsilon_explorations(
                    env, algo, eps, best_id=best_policy_id)
            
            l_actions.append(actions_one_game.reshape(actions_one_game.shape[0], 1))
            l_observations.append(observations_one_game)

            if self.multi_game:
                labels = np.full((actions_one_game.shape[0], 1), d_games[env])
            else:
                labels = np.full((actions_one_game.shape[0], 1), 0)

            l_game_labels.append(labels)

        actions = np.vstack(l_actions)
        actions = actions.reshape(actions.shape[0])
        observations = np.vstack(l_observations)
        game_labels = np.vstack(l_game_labels)
        game_labels = game_labels.reshape(game_labels.shape[0])

        return observations, actions, game_labels

    def generate_dataset(self, data_cases, algo):
        self.batch_counter = 0

        observations, actions, game_labels = self.generate_trajectories_for_given_cases(data_cases, algo)

        indices = np.random.permutation(actions.shape[0])
        self.observations = observations[indices]
        self.actions = actions[indices]
        
        #if not self.multi_game:
        #    return observations, actions
        
        self.game_labels = game_labels[indices]

        print(f"Nr of batches is {actions.shape[0]}")
        self.nr_of_batches = actions.shape[0]

    def get_next_batch_if_possible(self, batch_size):
        if self.batch_counter + batch_size >= self.nr_of_batches:
            return None, True

        obs = self.observations[self.batch_counter: self.batch_counter + batch_size]
        act = self.actions[self.batch_counter: self.batch_counter + batch_size]

        self.batch_counter += batch_size

        return obs, act, False


def get_epsilons_for_explorations():
    spaces = [ 
        np.linspace(0.0, 0.1, num=300),
        np.linspace(0.1, 0.15, num=50]
    ]
    epsilons = np.concatenate(spaces)
    
    return epsilons

class MultiDatasetLoader():

    def __init__(self, envs, algo, multi_game, nr_policy, seed, saved_trajectories = "./saved_trajectories"):
        self.data_creator = ChunksCreator(envs, nr_policy, seed, multi_game=multi_game)
        self.saved_trajectories = saved_trajectories
        self.algo = algo

        self.epsilons = get_epsilons_for_explorations()
        self.all_data_cases = np.array([(game, eps) for eps in self.epsilons for game in envs])

        np.random.shuffle(self.all_data_cases)

        self.it = 1
        self.trajectories_at_once = 10
        self.generate_dataset(algo=algo, epsilons=self.epsilons)
        
    def generate_dataset(self, algo):
        if self.it * 10 > len(self.all_data_cases):
            print("End of data")
            return
        
        data_cases = self.all_data_cases[(self.it-1)*10: self.it*10]
        self.data_creator.generate_dataset(self.all_training_cases, self.algo)
        utils.fix_path()

        self.it += 1

    def get_next_batch_if_possible(self, batch_size):
        return self.data_creator.get_next_batch_if_possible(batch_size)
