from copy import deepcopy

from es import OpenES
from munch import Munch

NOT_SET = 'not_set'
DEFAULT_CONFIG = Munch(
  exp_name=NOT_SET,
  env_name=NOT_SET,
  perturbation_size=NOT_SET,
  episode_len_limit=20000,
  img_size=84,
  algo='rainbow',
  atari_zoo_run_id=1,
  atari_zoo_tag='final',
  num_workers=40,
  evolution_algo=OpenES,
  evolution_algo_kwargs=Munch(
    popsize=200,
    weight_decay=0.,
  ),
  neptune_project='michalzajac/adversarials-evolution',
  num_iterations=1e9,
)


def recurrent_update(target, source):
  for k, v in source.items():
    if k in target and isinstance(v, dict):
      recurrent_update(target[k], v)
    else:
      target[k] = v


def get_config(cfg):
  result = deepcopy(DEFAULT_CONFIG)
  recurrent_update(result, cfg)
  for v in result.values():
    assert v != NOT_SET
  return result
