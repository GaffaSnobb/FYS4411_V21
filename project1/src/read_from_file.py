import numpy as np
import matplotlib.pyplot as plt

def read_from_file():
    alpha, var, exp = np.loadtxt(fname="generated_data/output.txt", skiprows=1, unpack=True)
    plt.plot(alpha, exp)
    plt.xlabel("alpha")
    plt.ylabel("energy")
    plt.show()

def read_from_file_v2():
    alpha_impor, var_impor, exp_impor = np.loadtxt(fname="generated_data/output_importance.txt", skiprows=1, unpack=True)
    alpha_brute, var_brute, exp_brute = np.loadtxt(fname="generated_data/output_bruteforce.txt", skiprows=1, unpack=True)

    plt.figure()
    plt.plot(alpha_impor, exp_impor, label="importance")
    plt.plot(alpha_brute, exp_brute, label="brute force")
    plt.xlabel("alpha")
    plt.ylabel("energy")
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(alpha_impor, exp_impor-exp_brute, label="difference")
    # plt.xlabel("alpha")
    # plt.ylabel("energy")
    # plt.show()



if __name__ == "__main__":
    #read_from_file()
    read_from_file_v2()
