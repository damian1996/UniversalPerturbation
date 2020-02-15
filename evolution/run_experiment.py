import os
import signal
import time
from multiprocessing import Process, Queue

import neptune
import numpy as np
import tensorflow as tf
from atari_zoo import MakeAtariModel
from lucid.optvis.render import import_model

from configs import get_config
from utils import create_env, log_metric, Saver


def process_perturbation(config, x):
  x = np.clip(x, -1., 1.)
  x *= config.perturbation_size
  x = x.reshape([config.img_size, config.img_size, 1])
  return x


def evaluate_population(config, tasks_q, results_q, perturbations):
  returns = [None] * len(perturbations)
  for i, p in enumerate(perturbations):
    tasks_q.put((p, i))
  for _ in range(len(perturbations)):
    r, id = results_q.get()
    returns[id] = r
  return np.array(returns)


def run_worker(config, tasks_q, results_q):
  m = MakeAtariModel(config.algo, config.env_name, config.atari_zoo_run_id,
                     config.atari_zoo_tag)()
  m.load_graphdef()
  env = create_env(m)

  with tf.Session() as sess:
    input_ph = tf.placeholder(tf.float32,
                              [None] + list(env.observation_space.shape))
    imported_model = import_model(m, input_ph, input_ph)
    action_tensor = m.get_action(imported_model)

    while True:
      obs = env.reset()
      perturbation, id = tasks_q.get()
      result = 0
      steps = 0
      done = False
      while not done and steps < config.episode_len_limit:
        obs += perturbation
        action = sess.run(action_tensor, feed_dict={input_ph: obs[None]})[0]
        obs, reward, done, info = env.step(action)
        result += reward
        steps += 1
      results_q.put((result, id))


def run_experiment(config):
  config = get_config(config)

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGINT, original_sigint_handler)

  tasks_q, results_q = Queue(), Queue()
  ps = [Process(target=run_worker, args=(config, tasks_q, results_q)) for _ in
        range(config.num_workers)]
  for p in ps:
    p.start()

  try:
    solver = config.evolution_algo(
      num_params=config.img_size ** 2, **config.evolution_algo_kwargs)
    saver = Saver()

    with_neptune = False
    if config.neptune_project is not None:
      neptune.init(config.neptune_project)
      neptune.create_experiment(name=config.exp_name, params=config)
      with_neptune = True

    i = 0
    while i < config.num_iterations:
      start_time = time.time()

      population = solver.ask()
      perturbations = [process_perturbation(config, x) for x in population]
      returns = evaluate_population(config, tasks_q, results_q, perturbations)
      solver.tell(-returns)

      if with_neptune:
        saver.save(perturbations, returns)

      elapsed_time = time.time() - start_time
      log_metric(i, 'iteration_time', elapsed_time, with_neptune)
      log_metric(i, 'pop_score_min', returns.min(), with_neptune)
      log_metric(i, 'pop_score_mean', returns.mean(), with_neptune)
      log_metric(i, 'pop_score_max', returns.max(), with_neptune)
      i += 1

    if config.neptune_project is not None:
      neptune.stop()
  finally:
    for p in ps:
      p.terminate()
      p.join()
