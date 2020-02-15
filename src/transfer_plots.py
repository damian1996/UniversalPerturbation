import numpy as np
import cv2
import matplotlib.pyplot as plt

import measure_transfer_power as mtp
import utils

def get_transfer_power(results):
    return np.mean(results)

def plot_color_table(data, path):
    plt.figure(figsize=(10, 10))
    plt.ion()
    plt.set_cmap('bwr')
    c = plt.pcolor(data) #, edgecolors='k', linewidths=4, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(c)
    plt.savefig(path)
    plt.close()

def create_transfer_table(results):
    transfer_power = get_transfer_power(results)
    utils.fix_path()

    colormap = results.reshape(53, 53)
    print(colormap.shape)

    plot_color_table(colormap*256, "final_plots/test.png")
    # cv2.imwrite("test.png", colormap*256)
    return "test.png", transfer_power


# trained_policy_score = read_baselines_from_files("trained", all_envs, algo)
# random_policy_score = read_baselines_from_files("random", all_envs, algo)

# for mode in modes:
#     colormap = np.zeros((len(all_envs), len(all_envs)))
#     all_test_results = []

#     for i, env in enumerate(all_envs):
#         game = env.lower()

#         trained_pert_path = f"final_results/perts/{game}_{algo}_{mode}_{s_max_noise}_{s_lr}_{repeats}_{nr_test_runs}_single_best_policy.npy"
#         random_pert_path = f"final_results/perts/random_{s_max_noise}_pert.npy"

#         if mode == "trained":
#             perturbation = np.load(trained_pert_path)
#         elif mode == "random":
#             perturbation = np.load(random_pert_path)

#         for j, test_env in enumerate(all_envs):
#             test_env = f"{test_env}NoFrameskip-v4"
#             test_game = test_env.lower()
            
#             results2 = [0., 0., 0.]
#             for rep_id in range(rn_runs):
#                 results = perturbation_test.run_experiments_for_env(perturbation, test_env)
#                 results = [results[0][0], results[1][0], results[2][0]]
#                 for kk, res in enumerate(results):
#                     results2[kk] += res

#             results2 = np.array(results2) / nr_test_runs

#             normalized_result = normalizations.normalize(results2, random_policy_score[j], trained_policy_score[j])
            
#             print(results2)
#             print("Normalized", normalized_result)
#             colormap[i, j] = normalized_result

#             s = (f"{env} {test_env}"
#                 f" {str(results[0])} {str(results[1])} {str(results[2])}"
#                 f" {str(normalized_result)}"
#             )
#             print("Test", s)
#             all_test_results.append(s)
    

#     colormap_values_as_numpy_path = f"final_results/colormaps/{algo}_{mode}_{s_max_noise}_{s_lr}_{repeats}_{nr_test_runs}_single_best_policy.npy"
#     colormap_as_image_path = f"final_results/colormaps/{algo}_{mode}_{s_max_noise}_{s_lr}_{repeats}_{nr_test_runs}_single_best_policy.png"
#     all_results_for_case_path = f"final_results/all_results/{algo}_{mode}_{s_max_noise}_{s_lr}_{repeats}_{nr_test_runs}_single_best_policy.txt"

#     np.save(colormap_values_as_numpy_path, colormap)

#     plots.normalized_color_table(colormap, colormap_as_image_path)

#     with open(all_results_for_case_path, "w") as f:
#         f.write('\n'.join(all_test_results))

# noises_to_transfer_power = [s_max_noise, s_max_noise]
# results_to_transfer_power = [f"results_colormap_random_{s_max_noise}_best_policy_lr_{s_lr}_repeating.txt", f"results_colormap_trained_{s_max_noise}_best_policy_lr_{s_lr}_repeating.txt"]

# mtp.plot_transfer_tables(results_to_transfer_power, noises_to_transfer_power)}}