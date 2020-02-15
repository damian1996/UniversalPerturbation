import numpy as np

noise_shape = (84, 84, 1)

def generate_random_noise_positions(noise_percent):
    arr = np.zeros((noise_shape[0]*noise_shape[1]*noise_shape[2],))
    indices = np.random.choice(np.arange(arr.size), replace=False, size=int(arr.size * noise_percent))
    arr[indices] = 1

    return arr

def prepare_noise_for_obs(noise_interval, noise_percent):
    np.random.seed(seed=11)
    positions = generate_random_noise_positions(noise_percent).astype(np.int64)
    noise = np.random.uniform(noise_interval[0], noise_interval[1], positions.shape)
    for i in range(noise.size):
        if not positions[i]:
            noise[i] = 0.

    return noise.reshape(noise_shape)

def generate_random_perturbation(noise_percent):
    return prepare_noise_for_obs((-noise_percent, noise_percent), 1.0)