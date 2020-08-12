try:
	import moviepy.editor as mpy
	from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
except:
	print("Moviepy not installed, movie generation features unavailable.")

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import os
import random
from io import BytesIO
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
from lucid.misc.io.serialize_array import _normalize_array
import generate_rollout as g_r


from models import MakeAtariModel
import atari_zoo.atari_wrappers as atari_wrappers
from atari_zoo.dopamine_preprocessing import AtariPreprocessing as DopamineAtariPreprocessing	
from atari_zoo.atari_wrappers import FireResetEnv, NoopResetEnv, MaxAndSkipEnv,WarpFrameTF,FrameStack,ScaledFloatFrame
import random_perturbation_generator as rpg
from perturbation_test import run_experiments_for_env
from consts import *


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def args_initialization(args, test_eps, max_frames, min_frames, output, sticky_action_prob, render, streamline, verbose, observation_noise):
    if args==None:
        arg_dict = {
            'test_eps':test_eps,
            'max_frames':max_frames,
            'min_frames':min_frames,
            'output':output,
            'sticky_action_prob':sticky_action_prob,
            'render':render,
            'streamline':streamline,
            'verbose':verbose,
            'observation_noise': observation_noise
        }
        args = dotdict(arg_dict)

    return args

def create_env(m, preprocessing):
    if preprocessing == 'dopamine':
        env = gym.make(m.environment)
        if hasattr(env,'unwrapped'):
            env = env.unwrapped
            env = DopamineAtariPreprocessing(env)
            env = FrameStack(env, 4)
            env = ScaledFloatFrame(env,scale=1.0/255.0)
    elif preprocessing == 'np':
        env = gym.make(m.environment)
        env = atari_wrappers.wrap_deepmind(env, episode_life=False,preproc='np')
    else:
        env = gym.make(m.environment)
        env = atari_wrappers.wrap_deepmind(env, episode_life=False,preproc='tf')
    
    return env

def run_session(sess, obs, action_sample, high_level_rep, policy, X_t, streamline):
    train_dict = {X_t:obs[None]}
     
    if streamline:
        results = sess.run([action_sample], feed_dict=train_dict)
        act = results[0] #grab action
    else:
        results = sess.run([action_sample,high_level_rep, policy], feed_dict=train_dict)
        act = results[0] #grab action
        representation = results[1][0] #get high-level representation
    
    return act, representation
            
def model_and_args_initialization(args, test_eps, max_frames, min_frames, output, 
                                  sticky_action_prob, render, streamline, verbose, observation_noise):
    if args==None:
        arg_dict = {
            'test_eps':test_eps,
            'max_frames':max_frames,
            'min_frames':min_frames,
            'output':output,
            'sticky_action_prob':sticky_action_prob,
            'render':render,
            'streamline':streamline,
            'verbose':verbose,
            'observation_noise': observation_noise
        }
        args = dotdict(arg_dict)
    
    return args

def model_initialization(model):
    m = model 
    preprocessing = m.preprocess_style
    m.load_graphdef()
    return m, preprocessing

def config_initialization(args):
    dev_cnt = 1
    if args.cpu:
        dev_cnt = 0
    dev_cnt = 0
    #for rollouts maybe don't use GPU?
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25),
        device_count = {'GPU': dev_cnt}
    )
    config.gpu_options.allow_growth=True

    return config

def draw_obs(obs, pert):
    w=4
    h=1
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 1
    # rand_pert = rpg.generate_random_perturbation(max_noise)
    magnitude = "0.005"
    obs_to_draw = np.transpose(obs, (2, 0, 1))

    if pert is None:
        plt.imshow(obs_to_draw[0])
    else:
        plt.imshow(obs_to_draw[0] + pert.reshape((84, 84)))
    
    # plt.axis('off')
    ax = plt.subplot(111)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    if pert is None:
        fig.suptitle("Without Perturbation", fontsize=28)#, fontweight='bold')
    else:
        fig.suptitle(f"Magnitude {magnitude}", fontsize=28)#, fontweight='bold')
    
    # for i in range(1, columns*rows +1):
    #     fig.add_subplot(rows, columns, i)
    #     # plt.imshow(rand_pert.reshape((84, 84)))
    #     plt.imshow(obs_to_draw[i-1] + pert.reshape((84, 84)))
    
    plt.savefig(f"final_plots/perturbed_obs_{magnitude}.png")
    # plt.show()
    print("DOOONEEE!")

def get_sampled_games():
    return [
        'BankHeist', 'Centipede', 'Phoenix', 'ChopperCommand', 'Gopher', 
        'Krull', 'YarsRevenge', 'Seaquest', 'Atlantis', 'Pong', 
        'Assault', 'Solaris', 'UpNDown', 'DoubleDunk', 'Breakout',
        'Tennis', 'StarGunner', 'Zaxxon', 'Qbert', 'Gravitar'
    ]# Breakout, Pong

def get_sampled_games_rainbow():
    return [
        'BankHeist', 'Centipede', 'Phoenix', 'ChopperCommand', 'Gopher',
        'Krull', 'YarsRevenge', 'Seaquest', 'Atlantis',
        'Assault', 'Solaris', 'UpNDown', 'DoubleDunk',
        'Tennis', 'StarGunner', 'Zaxxon', 'Qbert', 'Gravitar'
    ]# Breakout, Pong

def get_all_games_for_algo(algo):
    if algo in ["dqn", "rainbow"]:
        return full_dopamine_games_list
    elif algo in ["a2c", "impala", "ga"]:
        return full_a2c_games_list
    elif algo in ["es", "apex"]:
        return full_apex_games_list

def get_games_for_algo(algo):
    return get_sampled_games()
    # return get_all_games_for_algo(algo)

def normalize(results, random_score, trained_score):
    x = (results[0] + results[1] + results[2]) // 3

    if (random_score == trained_score):
        return 0

    score = (x - random_score) / (trained_score - random_score)

    if score < 0:
        score = 0
    elif score > 1:
        score = 1

    return score

def normalize2(results, random_score, trained_score, game):
    all_results = []
    for i in range(3):
        x = results[0][i]
        if random_score[i] == trained_score[i]:
            score = 0.0
        else:
            score = (x - random_score[i]) / (trained_score[i] - random_score[i])
        
        if score < 0:
            score = 0
        elif score > 1:
            score = 1
        
        all_results.append(score)

    return np.array(all_results).sum() / 3

def normalize_results(results, random_score, trained_score):
    all_results = []
    for i in range(len(results)):
        x = results[i]
        if random_score[i] == trained_score[i]:
            score = 0.0
        else:
            score = (x - random_score[i]) / (trained_score[i] - random_score[i])
        
        if score < 0:
            score = 0
        elif score > 1:
            score = 1
        
        all_results.append(score)

    return all_results

def normalize_result_for_one_policy(results, random_score, trained_score, game):
    all_results = []
    for i in range(1):
        x = results[0][i]
        if random_score == trained_score:
            score = 0.0
        else:
            score = (x - random_score) / (trained_score - random_score)

        if score < 0:
            score = 0
        elif score > 1:
            score = 1
        
        all_results.append(score)

    return np.array(all_results).sum()
    
def sample_unique_games_for_algo(algo, N):
    games = get_games_for_algo(algo)
    indices = np.array(random.sample(range(0, len(games)), N))
    return np.array(games)[indices].tolist()

def get_best_policies_id_per_game(envs, nr_policy):
    return {envs[i]: nr_policy for i in range(len(envs))}

def obs_plus_rand_pert():
    max_noise = 0.00001
    rand_pert = rpg.generate_random_perturbation(max_noise)
    obs = np.load("./example_obs.npy")
    print(obs.shape)
    perturbed = np.clip(obs+rand_pert, 0, 1)

    cv.imwrite("PerturbedObservations.png", perturbed*256)  

def fix_path():
    if "src" not in os.getcwd():
        os.chdir('src')
    elif "atari-model-zoo" in os.getcwd():
        os.chdir("..")

'''
def get_best_result_id_from_log(line):
    parts = line.split()
    #print([float(parts[2]), float(parts[3]), float(parts[4])])
    return np.argmin(np.array([float(parts[2]), float(parts[3]), float(parts[4])])) + 1

def get_best_policies_id_per_game():
    with open("colormaps/results_colormap_trained_0_008.txt", "r") as f:
        lines = [line.strip().lower() for line in f.readlines()] 
        
        best_policies_id = []
        for i, env in enumerate(all_envs):
            for line in lines:
                if line.startswith(f"{env} {env}".lower()):
                    best_id = get_best_result_id_from_log(line)

                    best_policies_id.append(best_id)
        
        envs_as_keys = [f"{env[0].capitalize()}{env[1: len(env)]}NoFrameskip-v4" for env in all_envs] 
        best_policies_dict = {envs_as_keys[i]: best_policies_id[i] for i in range(len(envs_as_keys))}

        return best_policies_dict
'''

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_computation_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    
    display(HTML(iframe))


def make_video_for_game(env, game, algo, noise_percent=0.0, fps=60.0, skip=1, ptype=None, video_fn='./output.mp4', pert_path=None, pert=None, random=False):
    if random:
        pert = rpg.prepare_noise_for_obs((-noise_percent, noise_percent), 1.0)
    else:
        if pert is None and pert_path is not None:
            pert = np.load(pert_path)

    for run_id in range(1,4):
        changed_noise_percent = str(noise_percent).replace(".", "_")
        video_fn = f"{game}_{run_id}_{algo}_{changed_noise_percent}_{ptype}.mp4" 
        video_fn = os.path.join("videos", video_fn)

        if os.path.exists(video_fn):
            print("Given file already exists")
            continue
        
        m = MakeAtariModel(algo, env, run_id, tag="video")()

        if pert is None:
            print("PERT IS NONE")
            results = g_r.generate_clean_rollout(m, max_frames=2500, min_frames=2500)
        else:
            results = g_r.generate_clean_rollout(m, max_frames=2500, min_frames=2500, perturbation=pert)

        obs = np.array(results['observations'])
        frames = np.array(results['frames'])
        size_x,size_y = frames.shape[1:3]

        writer = FFMPEG_VideoWriter(video_fn, (size_y, size_x), fps)
        for x in range(0,frames.shape[0],skip):
            writer.write_frame(frames[x])
        writer.close()

        print("Video created!")


if __name__ == "__main__":
    print(sample_unique_games_for_algo("rainbow", 20))

    # algo, env, game = "rainbow", "BreakoutNoFrameskip-v4", "breakout"
    # random_pert = prepare_noise_for_obs((-0.01, 0.01), 1.0)
    # make_video_for_game(env, game, algo, ptype="random", pert=random_pert)

    algo, env, game = "dqn", "PhoenixNoFrameskip-v4", "Phoenix"
    random_pert = np.load("./final_results_0_policy/random/random_perts/pert_0_01_0.npy")
    make_video_for_game(env, game, algo, ptype="random", pert=random_pert)

    trained_pert = np.load("./final_results_0_policy/trained/rainbow/perts/Phoenix/0/0_01/pert.npy")
    make_video_for_game(env, game, algo, ptype="trained", pert=trained_pert)

    make_video_for_game(env, game, algo)
