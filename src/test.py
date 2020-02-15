import numpy as np

if __name__ == "__main__":
    path = "final_results_0_policy/trained/a2c/results/Pong/0/0_005/pong.npy" 
    print(np.load(path))
    
    path = "final_results_0_policy/random/a2c/results/BankHeist/0/0_005/pong.npy"
    print(np.load(path))
