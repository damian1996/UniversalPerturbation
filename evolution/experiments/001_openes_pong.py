import os

from munch import Munch

from run_experiment import run_experiment

config = Munch(
  exp_name=os.path.basename(__file__)[:-3],
  env_name='PongNoFrameskip-v4',
  perturbation_size=0.01,
  evolution_algo_kwargs=Munch(
    sigma_init=1.0,
    sigma_limit=0.1,
  ),
)

run_experiment(config)
