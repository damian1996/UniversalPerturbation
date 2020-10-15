import tensorflow as tf
import numpy as np
import lucid
from lucid.optvis.render import import_model
import psutil, sys

import adjust_shapes_for_model
import utils
#import visualize_computation_graph
import neptune_client
import perturbation_test
import random_perturbation_generator as rpg

left, right = -0.0005, 0.0005
left_clip, right_clip = -0.00005, 0.00005
noise_shape = [84, 84, 1]

#@profile
def train_universal_perturbation_from_random_batches(model, to_be_perturbated=False, args=None, test_eps=1, data_loader=None, n_repeats=1,
                                                    max_frames=2500, min_frames=2500, dataset=None, nr_batches=0, max_noise=0., game=None,
                                                    algo=None, rep_buffer=None, all_training_cases=None, output='', sticky_action_prob=0.0, 
                                                    observation_noise = 0.0, render=False, cpu=False, streamline=False, verbose=True,
                                                    seed=11, pert_for_next_epoch=None):
    
    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    sticky_action_prob = args.sticky_action_prob
    m, preprocessing = utils.model_initialization(model)
    adjust_shapes_for_model.fix_reshapes(m, algo)
    config = utils.config_initialization(args)

    left, right = -max_noise, max_noise
    left_clip, right_clip = 5 * (left / 10), 5 * (right / 10)
   
    use_new_loss = True

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n

        obs = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape), name='obs')
        
        if pert_for_next_epoch is not None:
            init_data = pert_for_next_epoch
        else:
            init_data = rpg.get_random_gaussian_noise(noise_shape, seed, left_clip, right_clip)
        
        #init_data = rpg.get_random_gaussian_noise(noise_shape, seed)
        universal_perturbation = tf.Variable(init_data, name='universal_perturbation')
        X_t = obs + universal_perturbation

        T = import_model(m, X_t, X_t)
        
        action_sample, policy = m.get_action(T)
        action_sample = tf.cast(action_sample, tf.float64)
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2]

        clean_act = tf.placeholder(tf.int32, [None], name="action")
        
        if use_new_loss:
            probs = tf.nn.softmax(policy)
            entropy = probs * utils.log2(probs)
            loss = tf.reduce_sum(entropy) / batch_size
        else:
            probs = tf.nn.softmax(policy)
            loss = -tf.losses.sparse_softmax_cross_entropy(labels=clean_act, logits=policy)
        
        train_op = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)
        
        clip_val = tf.clip_by_value(universal_perturbation, clip_value_min=left, clip_value_max=right)
        assign_op = universal_perturbation.assign(clip_val)

        init = tf.global_variables_initializer()
        sess.run(init)

        # for v in  tf.global_variables(): print(v)

        # graph_def = tf.get_default_graph().as_graph_def()
        # tmp_def = utils.rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
        # utils.show_computation_graph(tmp_def)

        obs_noise = env.reset()
        ep_count, frame_count = 0, 1
        
        all_losses = []
        results_during_training = []
        
        last_update = False
        training_finished = False
        read_only_one_batch = True
        it = 1
        while not last_update: #training_finished: 
            if args.render: 
                env.render()
            
            clean_obs, clean_act1 = rep_buffer.get_single_batch(dataset[0], dataset[1], None, 32)
            perturbated_obs_to_graph = (clean_obs + universal_perturbation.eval(session=sess))
            
            _, l1, probs1 = sess.run([train_op, loss, probs], feed_dict=train_dict)

            if use_new_loss:
                train_dict = {X_t: perturbated_obs_to_graph, obs: clean_obs}
                _, l1, probs1, policy1 = sess.run([train_op, loss, probs, policy], feed_dict=train_dict)
                
                if frame_count % 10 == 0:
                    for i in range(batch_size):
                        avg_prob = 1.0 / probs1[i].shape[0]
                        if np.max(probs1[i]) > (1.1 * avg_prob):
                            print(i, ' ', np.min(probs1[i]), avg_prob, np.max(probs1[i]))
                
                loss_name = "Logits loss"
            else:
                train_dict = {X_t: perturbated_obs_to_graph, clean_act: tuple(clean_act1), obs: clean_obs}
                _, l1, probs1, policy1 = sess.run([train_op, loss, probs, policy], feed_dict=train_dict)
                
                loss_name = "Actions loss"
            
            if frame_count % 100 == 0:
                print(f"Actions loss {l1}")

            all_losses.append(l1)
            print(f"New loss {l1} and probs {probs1.shape}") 

            if frame_count % 100 == 0:
                print("timecheck", frame_count) 
                print(dict(psutil.virtual_memory()._asdict()))
 
            clip_dict = {}
            sess.run([assign_op], feed_dict=clip_dict)
                
            if frame_count % rep_buffer.replay_after_batches == 0:
                if last_update: break
                
                dataset, last_update = data_loader.get_updated_buffer(it) 
                it += 1
                
                # print("Meantime test")
                results_to_log = perturbation_test.run_experiments_for_env(universal_perturbation.eval(), game, algo=algo, meantime_test=True)
                results_during_training.append(results_to_log)

                print("Average perturbation", np.sum(universal_perturbation.eval()) / np.prod(noise_shape))

            frame_count += 1
        
        return {'perturbation': universal_perturbation.eval(), 'losses': all_losses, "old_results": results_during_training}



def train_universal_perturbation_from_full_dataset(model, to_be_perturbated=False, args=None, test_eps=1, n_repeats=1, game=None,
                                                    max_frames=2500, min_frames=2500, dataset=None, max_noise=0.,
                                                    algo=None, output='', sticky_action_prob=0.0, loader=None,
                                                    observation_noise = 0.0, render=False, cpu=False, streamline=False, verbose=True,
                                                    seed=11, pert_for_next_epoch=None):
    
    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    sticky_action_prob = args.sticky_action_prob
    m, preprocessing = utils.model_initialization(model)
    adjust_shapes_for_model.fix_reshapes(m, algo)
    config = utils.config_initialization(args)

    left, right = -max_noise, max_noise
    left_clip, right_clip = 5 * (left / 10), 5 * (right / 10)

    after_nr_batches_print_results = 150

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n

        obs = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape), name='obs')
        if pert_for_next_epoch is not None:
            init_data = pert_for_next_epoch
        else:
            init_data = rpg.get_random_gaussian_noise(noise_shape, seed, left_clip, right_clip)
            
        universal_perturbation = tf.Variable(init_data, name='universal_perturbation')
        X_t = obs + universal_perturbation

        T = import_model(m, X_t, X_t)
        
        action_sample, policy = m.get_action(T)
        action_sample = tf.cast(action_sample, tf.float64)
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2]

        clean_act = tf.placeholder(tf.int32, [None], name="action")

        loss = -tf.losses.sparse_softmax_cross_entropy(labels=clean_act, logits=policy)
        train_op = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)
        
        clip_val = tf.clip_by_value(universal_perturbation, clip_value_min=left, clip_value_max=right)
        assign_op = universal_perturbation.assign(clip_val)

        init = tf.global_variables_initializer()
        sess.run(init)

        obs_noise = env.reset()
        ep_count, frame_count = 0, 1
        
        all_losses = []
        results_during_training = []
        
        while not loader.is_batch_pool_exhausted():
            if args.render: 
                env.render()
            
            #repeat_counter = 0
            #while repeat_counter < n_repeats: 
            clean_obs, clean_act1 = loader.get_single_batch()
            perturbated_obs_to_graph = (clean_obs + universal_perturbation.eval(session=sess))
            
            train_dict = {X_t: perturbated_obs_to_graph, clean_act: tuple(clean_act1), obs: clean_obs}
            
            _, l1 = sess.run([train_op, loss], feed_dict=train_dict)
            
            all_losses.append(l1)

            clip_dict = {}
            sess.run([assign_op], feed_dict=clip_dict)
                
            if frame_count % after_nr_batches_print_results == 0:
                results_to_log = perturbation_test.run_experiments_for_env(universal_perturbation.eval(), game)
                results_during_training.append(results_to_log)

                print("Average perturbation", np.sum(universal_perturbation.eval()) / np.prod(noise_shape))
                print(dict(psutil.virtual_memory()._asdict()))
 
            frame_count += 1
        
        print(f"All batches number is {frame_count}")
        return {'perturbation': universal_perturbation.eval(), 'losses': all_losses, "old_results": results_during_training}


#@profile
def generate_more_informative_dataset(model, args=None, test_eps=1, data_loader=None, max_frames=2500, min_frames=2500, game=None,
    algo=None, output='', sticky_action_prob=0.0, observation_noise = 0.0, render=False, cpu=False, streamline=False, verbose=True, seed=11):
    
    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    sticky_action_prob = args.sticky_action_prob
    m, preprocessing = utils.model_initialization(model)
    adjust_shapes_for_model.fix_reshapes(m, algo)
    config = utils.config_initialization(args)
 
    use_new_loss = True
    batch_size = 64

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n

        obs = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape), name='obs')

        T = import_model(m, obs, obs)
        
        action_sample, policy = m.get_action(T)
        action_sample = tf.cast(action_sample, tf.float64)
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2]

        probs = tf.nn.softmax(policy)
        
        clean_act = tf.placeholder(tf.int32, [None], name="action")
        
        init = tf.global_variables_initializer()
        sess.run(init)

        obs_noise = env.reset()
        ep_count, frame_count = 0, 1
        
        full_obs, full_act = [], []
        nr_of_good_elements = 0
        
        while True: 
            if args.render: 
                env.render()

            clean_obs1, clean_act1, is_new_dataset_needed = data_loader.get_next_batch_if_possible(batch_size)
            if is_new_dataset_needed:
                is_finished = data_loader.generate_dataset()
                if is_finished:
                    break
                
                clean_obs1, clean_act1, is_new_dataset_needed = data_loader.get_next_batch_if_possible(batch_size)

            train_dict = {obs: clean_obs1, clean_act: tuple(clean_act1)}
            probs1, policy1 = sess.run([probs, policy], feed_dict=train_dict)

            print("checkpoint ", nr_of_good_elements)

            for i in range(batch_size):
                avg_prob = 1.0 / probs1[i].shape[0]
                if np.max(probs1[i]) > (1.2 * avg_prob):
                    full_obs.append(clean_obs1[i])
                    full_act.append(clean_act1[i])
                    nr_of_good_elements += 1
                    #print(i, ' ', np.min(probs1[i]), avg_prob, np.max(probs1[i]))

            if frame_count % 100 == 0:
                print(dict(psutil.virtual_memory()._asdict()))
 
            frame_count += 1

        obs = np.array(full_obs)
        act = np.array(full_act)
        print(obs.shape, act.shape)
        np.save(open(f'bigger_data/obs_{game}.npy', 'wb'), obs)
        np.save(open(f'bigger_data/act_{game}.npy', 'wb'), act)


def train_universal_perturbation_from_random_batches(model, to_be_perturbated=False, args=None, test_eps=1, data_loader=None, n_repeats=1,
                                                    max_frames=2500, min_frames=2500, dataset=None, nr_batches=0, max_noise=0., game=None,
                                                    algo=None, rep_buffer=None, all_training_cases=None, output='', sticky_action_prob=0.0,
                                                    observation_noise = 0.0, render=False, cpu=False, streamline=False, verbose=True,
                                                    seed=11, pert_for_next_epoch=None):

    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    sticky_action_prob = args.sticky_action_prob
    m, preprocessing = utils.model_initialization(model)
    adjust_shapes_for_model.fix_reshapes(m, algo)
    config = utils.config_initialization(args)

    left, right = -max_noise, max_noise
    left_clip, right_clip = 5 * (left / 10), 5 * (right / 10)

    use_new_loss = True

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n

        obs = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape), name='obs')

        if pert_for_next_epoch is not None:
            init_data = pert_for_next_epoch
        else:
            init_data = rpg.get_random_gaussian_noise(noise_shape, seed, left_clip, right_clip)

        #init_data = rpg.get_random_gaussian_noise(noise_shape, seed)
        universal_perturbation = tf.Variable(init_data, name='universal_perturbation')
        X_t = obs + universal_perturbation

        T = import_model(m, X_t, X_t)

        action_sample, policy = m.get_action(T)
        action_sample = tf.cast(action_sample, tf.float64)
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2]

        clean_act = tf.placeholder(tf.int32, [None], name="action")
		
		if use_new_loss:
            probs = tf.nn.softmax(policy)
            entropy = probs * utils.log2(probs)
            loss = tf.reduce_sum(entropy) / batch_size
        else:
            probs = tf.nn.softmax(policy)
            loss = -tf.losses.sparse_softmax_cross_entropy(labels=clean_act, logits=policy)

        train_op = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

        clip_val = tf.clip_by_value(universal_perturbation, clip_value_min=left, clip_value_max=right)
        assign_op = universal_perturbation.assign(clip_val)

        init = tf.global_variables_initializer()
        sess.run(init)

        obs_noise = env.reset()
        ep_count, frame_count = 0, 1

        all_losses = []
        results_during_training = []

        read_only_one_batch = True
        it = 1
        while not data_loader.is_finished():
            if args.render:
                env.render()

            clean_obs, clean_act1 = data_loader.get_next_batch(128)
            perturbated_obs_to_graph = (clean_obs + universal_perturbation.eval(session=sess))

            _, l1, probs1 = sess.run([train_op, loss, probs], feed_dict=train_dict)

			if use_new_loss:
                train_dict = {X_t: perturbated_obs_to_graph, obs: clean_obs}
                _, l1, probs1, policy1 = sess.run([train_op, loss, probs, policy], feed_dict=train_dict)

                if frame_count % 10 == 0:
                    for i in range(batch_size):
                        avg_prob = 1.0 / probs1[i].shape[0]
                        if np.max(probs1[i]) > (1.2 * avg_prob):
                            print(i, ' ', np.min(probs1[i]), avg_prob, np.max(probs1[i]))

                loss_name = "Logits loss"
            else:
                train_dict = {X_t: perturbated_obs_to_graph, clean_act: tuple(clean_act1), obs: clean_obs}
                _, l1, probs1, policy1 = sess.run([train_op, loss, probs, policy], feed_dict=train_dict)

                loss_name = "Actions loss"

            if frame_count % 50 == 0:
                print(f"{loss_name} {l1}")

            if frame_count % 100 == 0:
                print("timecheck", frame_count)
                print(dict(psutil.virtual_memory()._asdict()))

            clip_dict = {}
            sess.run([assign_op], feed_dict=clip_dict)

            if frame_count % 150 == 0:
                results_to_log = perturbation_test.run_experiments_for_env(universal_perturbation.eval(), game, algo=algo, meantime_test=True)
                results_during_training.append(results_to_log)

            frame_count += 1

        return {'perturbation': universal_perturbation.eval(), 'losses': all_losses, "old_results": results_during_training}

