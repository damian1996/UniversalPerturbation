import pickle
import tempfile

import atari_zoo.atari_wrappers as atari_wrappers
import gym
import neptune
import numpy as np
from atari_zoo.atari_wrappers import FrameStack, ScaledFloatFrame
from atari_zoo.dopamine_preprocessing import \
  AtariPreprocessing as DopamineAtariPreprocessing


def create_env(model):
  preprocessing = model.preprocess_style
  if preprocessing == 'dopamine':
    env = gym.make(model.environment)
    if hasattr(env, 'unwrapped'):
      env = env.unwrapped
      env = DopamineAtariPreprocessing(env)
      env = FrameStack(env, 4)
      env = ScaledFloatFrame(env, scale=1.0 / 255.0)
  elif preprocessing == 'np':
    env = gym.make(model.environment)
    env = atari_wrappers.wrap_deepmind(env, episode_life=False, preproc='np')
  else:
    env = gym.make(model.environment)
    env = atari_wrappers.wrap_deepmind(env, episode_life=False, preproc='tf')

  return env


def log_metric(iteration, k, v, with_neptune):
  print('it {}, {}: {}'.format(iteration, k, v))
  if with_neptune:
    neptune.log_metric(k, v)


def _save(perturbation, filename):
  with tempfile.NamedTemporaryFile() as temp:
    pickle.dump(perturbation, temp)
    neptune.send_artifact(temp.name, filename)


class Saver:
  def __init__(self):
    self.best_mean = float('inf')
    self.best_min = float('inf')

  def save(self, perturbations, returns):
    pert = perturbations[np.argmin(returns)]
    if self.best_mean > returns.mean():
      self.best_mean = returns.mean()
      _save(pert, 'best_mean')
    if self.best_min > returns.min():
      self.best_min = returns.min()
      _save(pert, 'best_min')
