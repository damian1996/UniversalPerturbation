import numpy as np
import tensorflow as tf
from numpy.random import default_rng
import os

from consts import *
import utils
import actions_and_obs_provider

class ReplayBuffer:

    def __init__(self, envs, nr_policy, seed, multi_game=False):
        self.trajectories_at_once = 20
        self.nr_new_trajectories = 5
        self.replay_after_batches = 120

        self.cnt_trajectories = self.trajectories_at_once
        self.rng = default_rng(seed)
        self.multi_game = multi_game
        self.d_games = {env: i for i, env in enumerate(envs)}

        self.best_policies_id = utils.get_best_policies_id_per_game(envs, nr_policy) 
        print("ONE BEST POLICY TO RULE THEM ALL", self.best_policies_id)

    def generate_trajectories_for_given_cases(self, initial_cases, algo):
        print(initial_cases)
        
        print(self.cnt_trajectories, "CNT TRAJECTORIES")
        l_actions, l_observations, l_game_labels = [], [], []

        for i, (env, eps) in enumerate(initial_cases): 
            best_id = self.best_policies_id[env] 
            observations_one_game, actions_one_game, _ = actions_and_obs_provider.get_all_actions_and_observations_with_epsilon_explorations(
                    env, algo, eps, best_id=best_id)
            
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

    def generate_initial_dataset(self, all_training_cases, algo):
        initial_cases = all_training_cases[:min(self.trajectories_at_once, len(all_training_cases))]
        observations, actions, game_labels = self.generate_trajectories_for_given_cases(initial_cases, algo)
         
        indices = np.random.permutation(actions.shape[0])
        actions = actions[indices]
        observations = observations[indices]
        #if not self.multi_game:
        #    return observations, actions
         
        game_labels = game_labels[indices]         

        return observations, actions, game_labels
   
    def get_single_batch(self, observations, actions, labels, batch_size): 
        numbers = self.rng.choice(actions.shape[0], size=batch_size, replace=False)

        if labels is None:
            return observations[numbers], actions[numbers]
        
        return observations[numbers], actions[numbers], labels[numbers]  
    
    def update_counter(self):
        self.cnt_trajectories += self.nr_new_trajectories

    def get_new_cases(self, all_training_cases):
        return all_training_cases[self.cnt_trajectories: min(self.cnt_trajectories+self.nr_new_trajectories, len(all_training_cases))]


    def generate_new_batches(self, all_training_cases, dataset_size, old_dataset, algo, temp_dataset=None):
        is_last_change = True
        
        print(all_training_cases)
        if self.cnt_trajectories < len(all_training_cases):
            is_last_change = False
            new_cases = all_training_cases[self.cnt_trajectories: min(
                self.cnt_trajectories+self.nr_new_trajectories, len(all_training_cases))]
        
            if not temp_dataset:
                observations, actions, game_labels = self.generate_trajectories_for_given_cases(new_cases, algo)
                numbers = self.rng.choice(dataset_size, size=actions.shape[0], replace=False)

                for i, nr in enumerate(numbers):
                    old_dataset[0][nr] = observations[i]
                    old_dataset[1][nr] = actions[i]
                    if old_dataset[2] is not None:
                        old_dataset[2][nr] = game_labels[i]
            else:
                numbers = self.rng.choice(dataset_size, size=temp_dataset[1].shape[0], replace=False)

                for i, nr in enumerate(numbers):
                    old_dataset[0][nr] = temp_dataset[0][i]
                    old_dataset[1][nr] = temp_dataset[1][i]
                    if old_dataset[2] is not None:
                        old_dataset[2][nr] = temp_dataset[2][i]
            

            self.cnt_trajectories += self.nr_new_trajectories
            print("Next trajectory", self.cnt_trajectories)

        return old_dataset, is_last_change


def get_epsilons_for_explorations(sizes):
    spaces = [ 
        np.linspace(0.0, 0.1, num=sizes[0]), # 60
        np.linspace(0.1, 0.15, num=sizes[1])[1:] # 9
    ]
    epsilons = np.concatenate(spaces)
    return epsilons

class TrajectoriesLoader():

    def __init__(self, envs, algo, multi_game, nr_policy, seed, saved_trajectories = "./saved_trajectories"):
        self.rep_buffer = ReplayBuffer(envs, nr_policy, seed, multi_game=multi_game)
        self.saved_trajectories = saved_trajectories

        self.epsilons = get_epsilons_for_explorations((60, 9))
        self.all_training_cases = np.array([(game, eps) for eps in self.epsilons for game in envs])
        print(self.epsilons)

        np.random.shuffle(self.all_training_cases)

        self.it = 1
        self.save_buffer_in_parts(algo=algo, epsilons=self.epsilons)
        
    def save_buffer_in_parts(self, algo=None, epsilons=None):
        # save initial buffer
        observations, actions, game_labels = self.rep_buffer.generate_initial_dataset(self.all_training_cases, algo)
        utils.fix_path()
        
        if not os.path.exists(self.saved_trajectories):
            os.mkdir(self.saved_trajectories) 
        
        np.save(open(f'{self.saved_trajectories}/obs0.npy', 'wb'), observations)
        np.save(open(f'{self.saved_trajectories}/act0.npy', 'wb'), actions)
        np.save(open(f'{self.saved_trajectories}/labels0.npy', 'wb'), game_labels)

        # save updated buffers
        dataset = (observations, actions, game_labels)
        
        while True:
            print(f"Iteration {self.it}") 
            new_cases = self.rep_buffer.get_new_cases(self.all_training_cases)
            
            if len(new_cases) == 0:
                break

            observations, actions, game_labels = self.rep_buffer.generate_trajectories_for_given_cases(new_cases, algo) 
            
            np.save(open(f'{self.saved_trajectories}/obs{self.it}.npy', 'wb'), observations)
            np.save(open(f'{self.saved_trajectories}/act{self.it}.npy', 'wb'), actions)
            np.save(open(f'{self.saved_trajectories}/labels{self.it}.npy', 'wb'), game_labels)
            
            self.it += 1
            
            self.rep_buffer.update_counter()

    def get_initial_buffer(self):
        return (np.load(f"{self.saved_trajectories}/obs0.npy"), np.load(f"{self.saved_trajectories}/act0.npy"), np.load(f"{self.saved_trajectories}/labels0.npy")), False

    def get_updated_buffer(self, it):
        last_update = (it == (self.it-1))
        return (np.load(f"{self.saved_trajectories}/obs{it}.npy"), np.load(f"{self.saved_trajectories}/act{it}.npy"), np.load(f"{self.saved_trajectories}/labels{it}.npy")), last_update


class DatasetInChunksLoader():

    def __init__(self, envs, algo, multi_game, nr_policy, seed, saved_trajectories = "./saved_trajectories"):
        self.rep_buffer = ReplayBuffer(envs, nr_policy, seed, multi_game=multi_game)
        self.saved_trajectories = saved_trajectories

        self.epsilons = get_epsilons_for_explorations((60, 9))
        self.all_training_cases = np.array([(game, eps) for eps in self.epsilons for game in envs])
        print(self.epsilons)

        np.random.shuffle(self.all_training_cases)

        self.it = 1
        self.save_buffer_in_parts(algo=algo, epsilons=self.epsilons)
        
    def save_buffer_in_parts(self, algo=None, epsilons=None):
        # save initial buffer
        observations, actions, game_labels = self.rep_buffer.generate_initial_dataset(self.all_training_cases, algo)
        utils.fix_path()
        
        if not os.path.exists(self.saved_trajectories):
            os.mkdir(self.saved_trajectories) 
        
        np.save(open(f'{self.saved_trajectories}/obs0.npy', 'wb'), observations)
        np.save(open(f'{self.saved_trajectories}/act0.npy', 'wb'), actions)
        np.save(open(f'{self.saved_trajectories}/labels0.npy', 'wb'), game_labels)

        # save updated buffers
        dataset = (observations, actions, game_labels)
        
        while True:
            print(f"Iteration {self.it}") 
            new_cases = self.rep_buffer.get_new_cases(self.all_training_cases)
            
            if len(new_cases) == 0:
                break

            observations, actions, game_labels = self.rep_buffer.generate_trajectories_for_given_cases(new_cases, algo) 
            
            np.save(open(f'{self.saved_trajectories}/obs{self.it}.npy', 'wb'), observations)
            np.save(open(f'{self.saved_trajectories}/act{self.it}.npy', 'wb'), actions)
            np.save(open(f'{self.saved_trajectories}/labels{self.it}.npy', 'wb'), game_labels)
            
            self.it += 1
            
            self.rep_buffer.update_counter()

    def get_initial_buffer(self):
        return (np.load(f"{self.saved_trajectories}/obs0.npy"), np.load(f"{self.saved_trajectories}/act0.npy"), np.load(f"{self.saved_trajectories}/labels0.npy")), False

    def get_updated_buffer(self, it):
        last_update = (it == (self.it-1))
        return (np.load(f"{self.saved_trajectories}/obs{it}.npy"), np.load(f"{self.saved_trajectories}/act{it}.npy"), np.load(f"{self.saved_trajectories}/labels{it}.npy")), last_update


class FullDatasetLoader():

    def __init__(self, envs, batch_size, algo, seed, multi_game=False):
        self.rng = default_rng()
        self.multi_game = multi_game
        self.batch_size = batch_size
        self.envs = envs

        self.d_games = {env: i for i, env in enumerate(envs)}
        self.best_policies_id = utils.get_best_policies_id_per_game(envs) 
        print("ONE BEST POLICY TO RULE THEM ALL", self.best_policies_id)

        self.observations, self.actions = self.generate_initial_dataset(algo)
        self.nr_batches = self.actions.shape[0] // self.batch_size
        print("Number of random steps", self.actions.shape[0])

        self.batch_cnt = 0
                                    
    def generate_training_cases(self, envs):
        epsilons = get_epsilons_for_explorations((45, 6))
        return np.array([(game, eps) for eps in epsilons for game in envs])
    
    def generate_trajectories_for_given_cases(self, all_training_cases, algo):
        l_actions, l_observations, l_game_labels = [], [], []
        for i, (env, eps) in enumerate(all_training_cases): 
            best_id = self.best_policies_id[env] 
            observations_one_game, actions_one_game, _ = actions_and_obs_provider.get_all_actions_and_observations_with_epsilon_explorations(
                    env, algo, eps, best_id=best_id)
            
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

    def generate_initial_dataset(self, algo):
        all_training_cases = self.generate_training_cases(self.envs)
        observations, actions, game_labels = self.generate_trajectories_for_given_cases(all_training_cases, algo)
         
        indices = np.random.permutation(actions.shape[0])
        actions = actions[indices]
        observations = observations[indices]
         
        # game_labels = game_labels[indices]         

        return observations, actions #, game_labels

    def is_batch_pool_exhausted(self):
        return not (self.batch_cnt < self.nr_batches)
   
    def get_single_batch(self):
        self.batch_cnt += 1
        numbers = self.rng.choice(self.actions.shape[0], size=self.batch_size, replace=False)
 
        return self.observations[numbers], self.actions[numbers] #labels[numbers]


class FullBetterDatasetLoader():

    def __init__(self, envs, batch_size, algo, seed, multi_game=False):
        self.rng = default_rng()
        self.multi_game = multi_game
        self.batch_size = batch_size
        self.envs = envs
		
		self.env = envs[0]

        self.observations, self.actions = np.load(f"bigger_data/obs_{self.env}.npy"), np.load("bigger_data/act_{self.env}.npy")
       	indices = np.random.permutation(self.actions.shape[0])
        self.actions = self.actions[indices]
        self.observations = self.observations[indices] 

		self.nr_batches = self.actions.shape[0] // self.batch_size

        self.batch_cnt = 0
	
	def is_finished(self):
		return not (self.batch_cnt < self.nr_batches)

	def get_next_batch(self, batch_size):
		obs = self.observations[self.batch_cnt * batch_size: (self.batch_cnt + 1) * batch_size]
        act = self.actions[self.batch_cnt * batch_size: (self.batch_cnt + 1) * batch_size]
		self.batch_cnt += 1

		return obs, act 
