import os
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')

from eval import eval_perturbation

game_name = "krull"
algo = "rainbow"
policy_id = 1
random_act = False
normalize_or_not=True
nr_runs = 1
perturbation = None

results = eval_perturbation(
    game_name, 
    algo, 
    policy_id, 
    perturbation=perturbation,
    normalize_or_not=normalize_or_not,
    random_act=random_act
)

print("FINAL PATH", os.getcwd())