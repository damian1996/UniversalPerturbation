import neptune
from consts import *
import numpy as np
import os
import collections
from operator import attrgetter

#TODO zrobić coś z games z consts.py

def save_games_results_in_the_middle_for_single_game(results_to_log, game_name):
    for jj in range(3):
        if jj == 0:
            neptune.log_metric(f"{game_name} & run {jj+1} & not seen", results_to_log[jj][0])
        else:
            neptune.log_metric(f"{game_name} & run {jj+1} & seen", results_to_log[jj][0])

def save_games_results_in_the_middle_for_multi_games(results_to_log):
    for ii in range(len(games)):
        for jj in range(3):
            if jj == 0:
                for kk in range(len(results_to_log[ii][jj])):
                    neptune.log_metric(f"{games[ii]} & run {jj+1} & not seen", results_to_log[ii][jj][kk])
            else:
                for kk in range(len(results_to_log[ii][jj])):
                    neptune.log_metric(f"{games[ii]} & run {jj+1} & seen", results_to_log[ii][jj][kk])

def neptune_backup_single_game_training(batch_size, lr, results_to_log, results_noise, noise_percent, game_name):
    neptune.init('damian1996/sandbox')
    
    hyperparameters = {'batch_size': batch_size, 'lr': lr, 'trajectories_per_game': 39}
    neptune.create_experiment(name=f"single_game_{noise_percent}", params=hyperparameters)

    neptune.log_text("Experiment description", f"Used games: {game_name}")
    
    for old_result in results_noise["old_results"]:
        save_games_results_in_the_middle_for_single_game(old_result, game_name)

    for jj in range(3):
        if jj == 0:
            neptune.log_metric(f"{game_name} & run {jj+1} & not seen", results_to_log[jj][0])
        else:
            neptune.log_metric(f"{game_name} & run {jj+1} & seen", results_to_log[jj][0])

    for val_loss in results_noise["losses"]:
        neptune.log_metric(f'loss_{game_name}', val_loss)

    with open('pert.npy', 'wb') as f:
        np.save(f, results_noise["perturbation"])

    neptune.log_artifact("pert.npy")

def neptune_backup_multi_game_training(batch_size, lr, results_to_log, results_noise, noise_percent, notseen):
    neptune.init('damian1996/sandbox')
    
    hyperparameters = {'batch_size': batch_size, 'lr': lr, 'trajectories_per_game': 26}
    neptune.create_experiment(name=f"multi_game_{noise_percent}", params=hyperparameters)

    neptune.log_text("Experiment description", f"Used games: {' '.join(sorted(games))}")
    
    desc_tag = f"Used-games-{'-'.join(sorted(games))}"
    lr_to_tag = str(lr).replace(".", "-")
    lr_tag = f"lr-{lr_to_tag}"
    max_noise_to_tag = str(noise_percent).replace(".", "-")
    max_noise_tag = f"max-noise-{max_noise_to_tag}"
    pert_size_tag = f"84-84-1"
    neptune.append_tag(desc_tag, lr_tag, max_noise_tag, pert_size_tag)

    for old_result in results_noise["old_results"]:
        save_games_results_in_the_middle_for_multi_games(old_result)
    
    for ii in range(len(games)):
        for jj in range(3):
            if jj == 0:
                for kk in range(len(results_to_log[ii][jj])):
                    neptune.log_metric(f"{games[ii]} & run {jj+1} & not seen", results_to_log[ii][jj][kk])
            else:
                for kk in range(len(results_to_log[ii][jj])):
                    neptune.log_metric(f"{games[ii]} & run {jj+1} & seen", results_to_log[ii][jj][kk])
    
    notseen_games = ["pong", "enduro"]
    for ii in range(len(notseen_games)):
        results_to_log = notseen[notseen_games[ii]]

        for jj in range(3):
            neptune.log_metric(f"{notseen_games[ii]} & run {jj+1} & not seen", results_to_log[jj][0])
    
    for idx, loss in enumerate(["losses1", "losses2"]): #, "losses3", "losses4"]):
        for val_loss in results_noise[loss]:
            neptune.log_metric(f'loss_{games[idx]}', val_loss)
 
    for idx in range(len(results_noise["losses1"])):
        neptune.log_metric(f'combined_losses', results_noise["losses1"][idx] + results_noise["losses2"][idx]) # + results_noise["losses3"][idx] + results_noise["losses4"][idx])

    with open('pert.npy', 'wb') as f:
        np.save(f, results_noise["perturbation"])

    neptune.log_artifact("pert.npy")

def save_tags_for_experiments(exps):
    for exp in exps:
        try:
            lr = str(exp.get_parameters()["lr"])
            desc = exp.get_logs()['Experiment description'].y
            if desc.startswith("Perturbation trained on only one game"):
                parts = desc.split(" ")
                desc = f"Used games: {parts[-1]}"
            
            lr = lr.replace(".", "-")
            lr = f"lr-{lr}"
            
            desc = desc.replace(" ", "-").replace(":", "")
            desc = f"{desc}"
            # for example desc='Used-games-breakout-seaquest'
            
            max_noise = exp.name.split("_")[-1].replace(".", "-")
            max_noise = f"max-noise-{max_noise}"
            
            exp.append_tag(desc, lr, max_noise)
            exp.remove_tag("0-001")
        except:
            None



def get_all_experiments_by_owner(project):
    return project.get_experiments(owner=['damian1996'])

def make_tags_backward_compatible(project):
    exps = get_all_experiments_by_owner(project)
    for exp in exps:
        tags = [tag for tag in exp.get_tags() if "Used" in tag]
        if len(tags) != 1:
            continue

        tag = "final"
        exp.remove_tag(tag)
        games = sorted(tag.split("-")[2:])
        new_tag = f"Used-games-{'-'.join(games)}"
        exp.append_tag(new_tag)

def get_experiments_for_lr_and_desc(project, lr, games, case=False):
    lr = str(lr).replace(".", "-")
    lr_tag = f"lr-{lr}"
    exps = project.get_experiments(tag=lr_tag)
    desc = f"Used-games-{'-'.join(sorted(games))}"
    desc_lower = f"used-games-{'-'.join(sorted(games))}"
    perturbation_shape = "84-84-1"

    exps2 = []
    for exp in exps:
        tags = exp.get_tags()
        if ((desc in tags) or (desc_lower in tags)):
            if not case:
                if (perturbation_shape not in tags):
                    exps2.append(exp)
            else:
                if (perturbation_shape in tags):
                    exps2.append(exp)

    return exps2

def get_experiments_with_learning_rates(project, games):
    Game = collections.namedtuple('Game', 'x y run_id noise_percent')
    desc_tag = f"Used-games-{'-'.join(sorted(games))}"
    exps = project.get_experiments(tag=desc_tag)
    desc_lower = f"used-games-{'-'.join(sorted(games))}"
    exps_lower = project.get_experiments(tag=desc_lower)
    exps.extend(exps_lower)

    perturbation_shape = "84-84-1"

    # for exp in exps:
    #     if (perturbation_shape not in exp.get_tags()):
    #         exps2.append(exp)

    lrs = np.unique([exp.get_parameters()["lr"] for exp in exps])
    print("lrs", lrs)
    exps_dict = {lr: [] for lr in lrs}
    for exp in exps:
        lr = exp.get_parameters()["lr"]
        exps_dict[lr].append(exp)
    
    return exps_dict

def get_experiments_lrs(games):
    api_token = os.environ['NEPTUNE_API_TOKEN']
    session = neptune.Session.with_default_backend(api_token=api_token)
    project = session.get_project('damian1996/sandbox')

    return get_experiments_with_learning_rates(project, games)

def get_experiments_games(lr, games, case=False):
    api_token = os.environ['NEPTUNE_API_TOKEN']
    session = neptune.Session.with_default_backend(api_token=api_token)
    project = session.get_project('damian1996/sandbox')

    return get_experiments_for_lr_and_desc(project, lr, games, case=case)

def sieve_same_experiments(exps):
    sieved = []
    results_sums = {}
    for i in range(0, len(exps), 3):
        if exps[i].noise_percent == "0.1" or exps[i].noise_percent == "0.05":
            continue
        
        key = exps[i].noise_percent
        sumi = exps[i].y + exps[i+1].y + exps[i+2].y

        if (key in results_sums) and (sumi > results_sums[key][0]) or (key not in results_sums):
            results_sums[key] = (sumi, i)
    
    sieved1 = [exps[i] for k, (v, i) in results_sums.items()]
    sieved2 = [exps[i+1] for k, (v, i) in results_sums.items()]
    sieved3 = [exps[i+2] for k, (v, i) in results_sums.items()]

    return (sieved1, sieved2, sieved3)

def parse_lists_of_experiments(exps, game):
    Game = collections.namedtuple('Game', 'x y run_id noise_percent')

    exps_for_game = []
    for exp in exps:
        for k, channel in exp.get_logs().items():
            if channel.channelType == 'numeric' and 'loss' not in channel.name:
                name = channel.name.split(' & ')[0]
                run_id = channel.name.split(' & ')[1]

                if name == game:
                    log = Game(x=channel.x, y=float(channel.y), run_id=run_id, noise_percent=exp.name.split('_')[-1])
                    exps_for_game.append(log)

    return exps_for_game

def parse_results_from_experiments(exps, game):
    exps_for_game = parse_lists_of_experiments(exps, game)

    return sieve_same_experiments(exps_for_game)

def parse_results_for_lrs_experiments(results, game):
    results2 = {}
    for k, exps in results.items():
        exps_for_game = parse_lists_of_experiments(exps, game)
        sieved1, sieved2, sieved3 = sieve_same_experiments(exps_for_game)
        results2[k] = (sieved1, sieved2, sieved3)
    
    return results2


# '''
# New way for neptune usage
# '''

# def backup_for_one_game_and_one_pert(batch_size=128, lr=0.001, results_for_all_games=None, 
#         nr_pert=0, perturbation=None, noise_size=0.0, game_name="", nr_trajectories=1): 

#     neptune.init('damian1996/PlotsVisualization')
    
#     hyperparameters = {
#         "batch_size": batch_size, 
#         "lr": lr, 
#         "nr_trajectories": nr_trajectories, 
#         "noise_size": noise_size
#     }
#     neptune.create_experiment(name=f"single_game_{noise_size}", params=hyperparameters)

#     neptune.log_text("Experiment description", f"Used games: {game_name}")
    
#     # neptune.log_metric(f"{game_name} & run {jj+1} & not seen", results_to_log[jj][0])

#     for old_result in results_noise["old_results"]:
#         save_games_results_in_the_middle_for_single_game(old_result, game_name)

#     for jj in range(3):
#         if jj == 0:
#             neptune.log_metric(f"{game_name} & run {jj+1} & not seen", results_to_log[jj][0])
#         else:
#             neptune.log_metric(f"{game_name} & run {jj+1} & seen", results_to_log[jj][0])

#     for val_loss in results_noise["losses"]:
#         neptune.log_metric(f'loss_{game_name}', val_loss)

#     with open('pert.npy', 'wb') as f:
#         np.save(f, results_noise["perturbation"])

#     neptune.log_artifact("pert.npy")