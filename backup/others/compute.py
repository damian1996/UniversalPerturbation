import sys
import numpy as np
import collections

def compute_variances(path):
    repeats = 5
    with open(path, "r") as f:
        lines = f.readlines()

        desc_lines = [line.strip()[18:].split(' ') for line in lines if line.startswith("Perturbation Test")]
        desc_cases = [(line[1], line[3].lower().split("no")[0]) for line in desc_lines]
        unique_desc_cases = [x for x in iter(collections.OrderedDict.fromkeys(desc_cases))]

        new_lines = [line.strip() for line in lines if line.startswith("No normalization")]

        boxes = []
        for i in range(0, len(new_lines), repeats):
            boxes.append(new_lines[i: i+repeats])

        for box_id, box in enumerate(boxes):
            p1, p2, p3 = [], [], []
            for results in box:
                results = results[18:len(results)-1]
                results = [float(result) for result in results.split(", ")]

                p1.append(results[0])
                p2.append(results[1])
                p3.append(results[2])

            p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

            print(f"Algorithm: {unique_desc_cases[box_id][0]}, Game: {unique_desc_cases[box_id][1]}")
            print(f"Policy 1: mean = {np.mean(p1)} , stddev = {np.std(p1)}, var = {np.var(p1)}")
            print(f"Policy 2: mean = {np.mean(p2)} , stddev = {np.std(p2)}, var = {np.var(p2)}")
            print(f"Policy 3: mean = {np.mean(p3)} , stddev = {np.std(p3)}, var = {np.var(p3)}")
            print()

def compute_means(path):
    repeats = 5
    with open(path, "r") as f:
        lines = f.readlines()

        desc_lines = [line.strip()[18:].split(' ') for line in lines if line.startswith("Perturbation Test")]
        desc_cases = [(line[1], line[3].lower().split("no")[0]) for line in desc_lines]
        unique_desc_cases = [x for x in iter(collections.OrderedDict.fromkeys(desc_cases))]

        new_lines = [line.strip() for line in lines if line.startswith("After normalization")]

        boxes = []
        for i in range(0, len(new_lines), repeats):
            boxes.append(new_lines[i: i+repeats])

        all_data = []
        for box_id, box in enumerate(boxes):
            p1 = []
            for results in box:
                results = float(results[20:len(results)])
                p1.append(results)
            
            mp1 = np.mean(np.array(p1))
            # print(unique_desc_cases[box_id], "  ", mp1)
            all_data.append((unique_desc_cases[box_id][0], unique_desc_cases[box_id][1], mp1))

        for i in range(0, len(all_data), 8):
            algo_data = all_data[i: i+8]
            m = np.mean(np.array([result for (algo, game, result) in algo_data]))

            print(algo_data[0][0], ' ', m)

# path = sys.argv[1]
# compute_variances(path)
# compute_means(path)