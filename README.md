# UniversalPerturbation


# Setup locally:
```
git clone https://github.com/damian1996/UniversalPerturbation.git

python3 -m venv univpert && source univpert/bin/activate

chmod +777 setup.sh && ./setup.sh

cd src

python3 main_multi_game.py or python3 main_single_game.py
```

#  Eval Script

cd src/

Usage
```
eval.py [-h] [--eps EPS] [--game GAME] [--random_act RANDOM_ACT] [--nr_runs NR_RUNS] perturbation_file
```

For example (in the example values for optional parameters are default values):
```
python3 eval.py perts/gopher_single_trained_0_01.npy --eps 0 --game pong --random_act False --nr_runs 1
```