import tensorflow as tf
import numpy as np
import lucid
from lucid.optvis.render import import_model
import psutil, sys

import adjust_shapes_for_model
import utils
import visualize_computation_graph
import neptune_client
import perturbation_test

left, right = -0.0005, 0.0005
left_clip, right_clip = -0.00005, 0.00005
noise_shape = [84, 84, 1]

def get_random_gaussian_noise(shape):
    total = np.prod(shape)
    init_data = np.random.normal(left_clip, right_clip, total).reshape(shape)
    convert_np_to_tf = tf.convert_to_tensor(init_data, dtype=tf.float32)
    return convert_np_to_tf.eval()

#@profile
def train_universal_perturbation_from_random_batches(model, to_be_perturbated=False, args=None, test_eps=1, data_loader=None,
                                                                    max_frames=2500, min_frames=2500, dataset=None, nr_batches=0, max_noise=0., game=None,
                                                                    algo=None, rep_buffer=None, all_training_cases=None, output='', sticky_action_prob=0.0, 
                                                                    observation_noise = 0.0, render=False, cpu=False, streamline=False, verbose=True):
    
    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    sticky_action_prob = args.sticky_action_prob
    m, preprocessing = utils.model_initialization(model)
    adjust_shapes_for_model.fix_reshapes(m, algo)
    config = utils.config_initialization(args)

    left, right = -max_noise, max_noise
    left_clip, right_clip = 5 * (left / 10), 5 * (right / 10)
   
    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n

        obs = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape), name='obs')
        init_data = get_random_gaussian_noise(noise_shape)
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

        # for v in  tf.global_variables(): print(v)

        # graph_def = tf.get_default_graph().as_graph_def()
        # tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
        # visualize_computation_graph.show_graph(tmp_def)

        obs_noise = env.reset()
        ep_count, frame_count = 0, 1
        
        all_losses = []
        results_during_training = []
        
        last_update = False
        training_finished = False
        read_only_one_batch = True
        it = 1
        while not training_finished:
            print("HAHAHHAHA")
            if args.render: 
                env.render()
             
            clean_obs, clean_act1 = rep_buffer.get_single_batch(dataset[0], dataset[1], None, 32)
            perturbated_obs_to_graph = (clean_obs + universal_perturbation.eval(session=sess))
            
            train_dict = {X_t: perturbated_obs_to_graph, clean_act: tuple(clean_act1), obs: clean_obs}
 
            _, l1 = sess.run([train_op, loss], feed_dict=train_dict)
            
            all_losses.append(l1)

            if frame_count % 100 == 0:
                print("timecheck", frame_count) 
                print(dict(psutil.virtual_memory()._asdict()))
 
            clip_dict = {}
            sess.run([assign_op], feed_dict=clip_dict)

            if frame_count % rep_buffer.replay_after_batches == 0:
                # if last_update: break
                
                # temp_dataset, last_update = data_loader.get_updated_buffer(it)
                # it += 1

                # dataset, training_finished = rep_buffer.generate_new_batches(all_training_cases, dataset[0].shape[0], dataset, algo, temp_dataset)
                
                if training_finished:
                    break
                dataset, training_finished = rep_buffer.generate_new_batches(all_training_cases, dataset[0].shape[0], dataset, algo)
                print(training_finished)

                results_to_log = perturbation_test.run_experiments_for_env(universal_perturbation.eval(), game)
                results_during_training.append(results_to_log)

                print("Average perturbation", np.sum(universal_perturbation.eval()) / np.prod(noise_shape))

            frame_count += 1
        
        return {'perturbation': universal_perturbation.eval(), 'losses': all_losses, "old_results": results_during_training}
