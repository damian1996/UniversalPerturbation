import os

from munch import Munch

from run_experiment import run_experiment

config = Munch(
  exp_name=os.path.basename(__file__)[:-3],
  env_name='PongNoFrameskip-v4',
  perturbation_size=0.01,
  num_workers=2,
  evolution_algo_kwargs=Munch(popsize=4),
  neptune_project=None,
  num_iterations=3,
)

run_experiment(config)
