import random as r
import tensorflow as tf
import numpy as np
import lucid
from lucid.optvis.render import import_model

import utils


def generate_clean_rollout(model,args=None,test_eps=1,max_frames=2500, min_frames=2500,
                           output='',sticky_action_prob=0.0, observation_noise = 0.0, random_act=False,
                           perturbation=None,render=False,cpu=False,streamline=False,verbose=True):

    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise) 
    m, preprocessing = utils.model_initialization(model)
    
    config = utils.config_initialization(args)

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n
        X_t = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))

        T = import_model(m,X_t,X_t)
        action_sample, policy = m.get_action(T)

        #get intermediate level representations
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2] # not output layer, but layer before

        sample_observations = []
        sample_frames = []
        sample_ram = []
        sample_representation = []
        sample_score = []
        sample_actions = []

        obs = env.reset()
        ep_count = 0
        rewards = []; ep_rew = 0.
        frame_count = 0
    
        prev_action = None

        # Evaluate policy over test_eps episodes
        while ep_count < args.test_eps: #or frame_count<=args.min_frames:
            if perturbation is not None:
                obs += perturbation
            
            if args.render:
                env.render()

            act, representation = utils.run_session(sess, obs, action_sample, high_level_rep, policy, X_t, streamline)

            if random_act:
                rand_act = np.random.randint(nA, size=1)
                act = [rand_act[0]]

            if not streamline:
                frame = env.render(mode='rgb_array')
                
                # print("Frame Count ", frame_count)
                # if (frame_count+1) % 300 == 0:
                #     print("LALA")
                #     # cv2_imshow(frame)
                #     utils.draw_obs(np.array(obs), perturbation)
                #     return

                sample_frames.append(np.array(frame,dtype=np.uint8))
                sample_ram.append(env.unwrapped._get_ram())
                sample_representation.append(representation)
                sample_observations.append(np.array(obs))

            #if frame_count == 246:
            #    np.save(open("./results_visualisation/example_obs.npy", "wb"), np.array(frame,dtype=np.uint8))

            sample_score.append(ep_rew)

            if prev_action != None and r.random() < sticky_action_prob:
                act = prev_action

            prev_action = act

            sample_actions.append(act)

            obs, rew, done, info = env.step(np.squeeze(act))
            ep_rew += rew
            frame_count += 1

            if frame_count >= args.max_frames:
                done=True

            if done:
                obs = env.reset()
                ep_count += 1
                rewards.append(ep_rew)
                ep_rew = 0.

        # if args.verbose:
        #     print(f"Avg. Episode Reward: {np.mean(rewards)}\n")
        #     print(f"Rewards: {rewards}\n")

        return {
            'actions': sample_actions,
            'observations':sample_observations,
            'frames':sample_frames,
            'ram':sample_ram,
            'representation':sample_representation,
            'score':sample_score,
            'ep_rewards':rewards
        }


"""Generate rollout with epsilon exploration"""

def rand_action_if_needed(curr_action, nr_actions, epsilon_for_explorations):
    if np.random.uniform(0,1,1)[0] < float(epsilon_for_explorations):
        return np.random.randint(nr_actions, size=1)[0]

    return curr_action

def generate_rollout_with_eps_explorations(model,args=None,test_eps=1,max_frames=2500, min_frames=2500,
                           output='',sticky_action_prob=0.0, observation_noise = 0.0,
                           perturbation=None, epsilon_for_explorations=0.0, random_act=False,
                           render=False,cpu=False,streamline=False,verbose=True):
    
    args = utils.args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise)
    m, preprocessing = utils.model_initialization(model)
    
    config = utils.config_initialization(args)

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = utils.create_env(m, preprocessing)
        nA = env.action_space.n
        X_t = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))

        T = import_model(m,X_t,X_t)
        action_sample, policy = m.get_action(T)

        #get intermediate level representations
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2] # not output layer, but layer before

        sample_observations = []
        sample_frames = []
        sample_ram = []
        sample_representation = []
        sample_score = []
        sample_actions = []

        obs = env.reset()
        ep_count = 0
        rewards = []; ep_rew = 0.
        frame_count = 0
    
        prev_action = None

        # Evaluate policy over test_eps episodes
        while ep_count < args.test_eps: #or frame_count<=args.min_frames:
            if perturbation is not None:
                obs += perturbation
            
            if args.render:
                env.render()

            act, representation = utils.run_session(sess, obs, action_sample, high_level_rep, policy, X_t, streamline)

            if not streamline:
                frame = env.render(mode='rgb_array')
                
                # print("Frame Count ", frame_count)
                # if (frame_count+1) % 400 == 0:
                #     # cv2_imshow(frame)
                #     utils.draw_obs(np.array(obs), perturbation)
                #     return

                sample_frames.append(np.array(frame,dtype=np.uint8))
                sample_ram.append(env.unwrapped._get_ram())
                sample_representation.append(representation)
                sample_observations.append(np.array(obs))
                
                #if frame_count == 246:
                #    np.save(open("./example_obs.npy", "wb"), representation)

            sample_score.append(ep_rew)

            if prev_action != None and r.random() < sticky_action_prob:
                act = prev_action

            prev_action = act

            act = rand_action_if_needed(np.squeeze(act), nA, epsilon_for_explorations)
            
            if random_act:
                rand_act = np.random.randint(nA, size=1)
                act = [rand_act[0]]

            sample_actions.append(np.array(act))

            obs, rew, done, info = env.step(act)
            ep_rew += rew
            frame_count += 1

            if frame_count >= args.max_frames:
                done=True

            if done:
                obs = env.reset()
                ep_count += 1
                rewards.append(ep_rew)
                ep_rew = 0.

        # if args.verbose:
        #     print(f"Avg. Episode Reward: {np.mean(rewards)}\n")
        #     print(f"Rewards: {rewards}\n")     

        return {
            'actions': sample_actions,
            'observations':sample_observations,
            'frames':sample_frames,
            'ram':sample_ram,
            'representation':sample_representation,
            'score':sample_score,
            'ep_rewards':rewards
        }
