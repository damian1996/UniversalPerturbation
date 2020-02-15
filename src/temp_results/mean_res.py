import numpy as np

# Krull, rozne seedy
print("Krull diff seeds")
krull_1_diff_seeds = [1497.0, 190.0, 2159.0, 3283.0, 3703.0, 6043.0, 6274.0, 6839.0, 1689.0, 5998.0]
print(np.mean(np.array(krull_1_diff_seeds)))
krull_2_diff_seeds = [8416.0, 7804.0, 5674.0, 8370.0, 4934.0, 4628.0, 4441.0, 5845.0, 5706.0, 5961.0]
print(np.mean(np.array(krull_2_diff_seeds)))
krull_3_diff_seeds = [4795.0, 2741.0, 9007.0, 610.0, 6225.0, 6064.0, 280.0, 2599.0, 6132.0, 6174.0]
print(np.mean(np.array(krull_3_diff_seeds)))
print()

# Krull, ten sam seed dla poczatkowych, rozne seedy w innych caseach
print("Krull same seeds")
krull_1_same_seeds = [430.0, 9512.0, 6430.0, 3512.0, 3067.0, 3433.0, 6502.0, 9925.0, 5959.0, 3380.0]
print(np.mean(np.array(krull_1_same_seeds)))
krull_2_same_seeds = [5369.0, 9133.0, 7052.0, 5921.0, 6636.0, 7862.0, 4374.0, 6367.0, 8728.0, 3821.0]
print(np.mean(np.array(krull_2_same_seeds)))
krull_3_same_seeds = [9444.0, 9402.0, 3396.0, 3524.0, 4149.0, 3171.0, 3773.0, 3885.0, 10013.0, 3563.0]
print(np.mean(np.array(krull_3_same_seeds)))
print()

# Phoenix, dqn, różne seedy
print("Phoenix diff seeds")
phoenix_1_diff_seeds = [5860.0, 5410.0, 100.0, 5050.0, 4550.0, 5490.0, 4820.0, 5220.0, 5120.0, 3600.0]
print(np.mean(np.array(phoenix_1_diff_seeds)))
phoenix_2_diff_seeds = [1070.0, 180.0, 2060.0, 380.0, 1440.0, 440.0, 320.0, 600.0, 260.0, 100.0]
print(np.mean(np.array(phoenix_2_diff_seeds)))
phoenix_3_diff_seeds = [1700.0, 3650.0, 120.0, 2110.0, 2250.0, 3720.0, 1570.0, 3510.0, 1660.0, 3220.0]
print(np.mean(np.array(phoenix_3_diff_seeds)))
print()

# Krull, diff seeds, random perts
print("Krull random perts")
krull_1_random = [490.0, 7206.0, 570.0, 4303.0, 4828.0, 4335.0, 3192.0, 8513.0, 3824.0, 4122.0]
print(np.mean(np.array(krull_1_random)))
krull_2_random = [280.0, 8980.0, 4535.0, 460.0, 5405.0, 6350.0, 3607.0, 880.0, 3707.0, 4040.0]
print(np.mean(np.array(krull_2_random)))
krull_3_random = [3678.0, 2993.0, 6311.0, 6309.0, 4319.0, 740.0, 4147.0, 4399.0, 4410.0, 6343.0]
print(np.mean(np.array(krull_3_random)))
print()

phoenix_1_random = [5200.0, 5310.0, 5440.0, 4870.0, 5220.0, 5310.0, 5720.0, 5920.0, 5160.0, 5020.0]
phoenix_2_random = [4890.0, 4500.0, 4990.0, 4990.0, 4660.0, 4770.0, 5380.0, 5390.0, 4580.0, 4480.0]
phoenix_3_random = [200.0, 5540.0, 2280.0, 3280.0, 4050.0, 3570.0, 1790.0, 5390.0, 2610.0, 4100.0]
print(np.mean(np.array(phoenix_1_random))) 
print(np.mean(np.array(phoenix_2_random)))
print(np.mean(np.array(phoenix_3_random)))

# Krull
# [4273.0]
# [4772.0]
# [6037.0]

# Phoenix
# [4520.0]
# [4870.0]
# [5140.0]