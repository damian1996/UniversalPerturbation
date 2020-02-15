import tensorflow as tf
import numpy as np
import lucid
from lucid.optvis.render import import_model
import psutil, sys

import adjust_shapes_for_model
import utils
import visualize_computation_graph
import perturbation_test
import neptune_client

left, right = -0.0005, 0.0005
left_clip, right_clip = -0.00005, 0.00005

def get_random_gaussian_noise(shape):
    total = np.prod(shape)
    init_data = np.random.normal(left_clip, right_clip, total).reshape(shape)
    convert_np_to_tf = tf.convert_to_tensor(init_data, dtype=tf.float32)
    return convert_np_to_tf.eval()

def cut_batch_to_smaller_parts(clean_obs, clean_act, labels, univ_pert, nr_models, counters):
    obs, act, perturbated_obs = [], [], [] 
    for i in range(nr_models):
        indices = np.where(labels == i)[0] 
        obs.append(clean_obs[indices])
        act.append(tuple(clean_act[indices], ))
        perturbated_obs.append(clean_obs[indices] + univ_pert)
        counters[i] += indices.shape[0] 

    return np.array(obs), np.array(act), np.array(perturbated_obs), counters

# @profile
def train_universal_perturbation_from_random_batches(models, to_be_perturbated=False, args=None, test_eps=1, max_frames=2500, 
                                                     min_frames=2500, dataset=None, nr_batches=0, max_noise=0., algo=None, lr=0.001, data_loader=None,
                                                     rep_buffer=None, all_training_cases=None, output='', sticky_action_prob=0.0, 
                                                     observation_noise = 0.0, render=False, cpu=False, streamline=False, verbose=True):
    
    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    sticky_action_prob = args.sticky_action_prob
    config = utils.config_initialization(args)
    
    # print(config)
    # config1 = tf.ConfigProto()
    # config1.gpu_options.allow_growth=True
    # print(config)
    #session = tf.Session(config=config) # use this for all sessions

    counters = [0 for el in models]

    ms = [] 
    for model in models:
        print(model)
        m, preprocessing = utils.model_initialization(model)
        adjust_shapes_for_model.fix_reshapes(m, algo)
        ms.append(m)

    left, right = -max_noise, max_noise
    left_clip, right_clip = 5 * (left / 10), 5 * (right / 10)

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        init_data = get_random_gaussian_noise([84, 84, 4])
        universal_perturbation = tf.Variable(init_data, name='universal_perturbation')

        sh = (84, 84, 4) 
        obss1 = tf.placeholder(tf.float32, [None] + list(sh), name=f"obss1")
        obss2 = tf.placeholder(tf.float32, [None] + list(sh), name=f"obss2")
        
        clean_actt1 = tf.placeholder(tf.int32, [None], name=f"action1")
        clean_actt2 = tf.placeholder(tf.int32, [None], name=f"action2")

        X_t1 = obss1 + universal_perturbation
        X_t2 = obss2 + universal_perturbation
        
        for idx, m in enumerate(ms):
            env = utils.create_env(m, preprocessing)
            nA = env.action_space.n

            if idx == 0:
                T = import_model(m, X_t1, X_t1, scope=f"import1")
            else:
                T = import_model(m, X_t2, X_t2, scope=f"import2")

            action_sample, policy = m.get_action(T)
            action_sample = tf.cast(action_sample, tf.float64)
            activations = [T(layer['name']) for layer in m.layers]
            high_level_rep = activations[-2]

            if idx == 0:
                loss1 = -tf.losses.sparse_softmax_cross_entropy(labels=clean_actt1, logits=policy)
            else:
                loss2 = -tf.losses.sparse_softmax_cross_entropy(labels=clean_actt2, logits=policy)
 
        train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss1 + loss2)
        
        clip_val = tf.clip_by_value(universal_perturbation, clip_value_min=left, clip_value_max=right)
        assign_op = universal_perturbation.assign(clip_val)

        init = tf.global_variables_initializer()
        sess.run(init) 

        obs_noise = env.reset()
        ep_count, frame_count = 0, 0
        
        losses1, losses2 = [], []
        results_during_training = []

        last_update = False
        training_finished = False
        read_only_one_batch = True
        it = 1
        while not last_update: 
            if args.render: 
                env.render()
             
            clean_obs, clean_act1, model_labels = rep_buffer.get_single_batch(dataset[0], dataset[1], dataset[2], 64)  
            
            univ_pert = universal_perturbation.eval(session=sess)
            arr_obs, arr_act, arr_pert_obs, counters = cut_batch_to_smaller_parts(clean_obs, clean_act1, model_labels, univ_pert, len(ms), counters)
            
            train_dict = {
                X_t1: arr_pert_obs[0], clean_actt1: np.array(arr_act[0]), obss1: arr_obs[0],
                X_t2: arr_pert_obs[1], clean_actt2: np.array(arr_act[1]), obss2: arr_obs[1]    
            }

            _, l1, l2 = sess.run([train_op, loss1, loss2], feed_dict=train_dict)
            losses1.append(l1)
            losses2.append(l2)
            
            if frame_count % 100 == 0:
                print("timecheck") 
                print(dict(psutil.virtual_memory()._asdict()))

            sess.run([assign_op], feed_dict={})

            if frame_count % rep_buffer.replay_after_batches == 0:
                if last_update: break

                temp_dataset, last_update = data_loader.get_updated_buffer(it)
                it += 1

                dataset, training_finished = rep_buffer.generate_new_batches(all_training_cases, dataset[0].shape[0], dataset, algo, temp_dataset=temp_dataset)
                
                results_to_log = perturbation_test.run_test_time_experiments(universal_perturbation.eval())
                results_during_training.append(results_to_log)

            frame_count += 1
        
        print("Class counters", ' '.join([str(el) for el in counters]))
        return {
            'perturbation': universal_perturbation.eval(),
            'losses1': losses1,
            'losses2': losses2,
            'old_results': results_during_training
        }
