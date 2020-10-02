import numpy as np
import os

files = [f"old_actions_loss_{i}.npy" for i in range(2, 50)]
print(files)

for filename in files[:1]:
    filename = filename.strip()
    print(f"perts_diffs/{filename}")
    arr = np.load(f"perts_diffs/{filename}")
    print(arr[0])
    print(arr[1])
