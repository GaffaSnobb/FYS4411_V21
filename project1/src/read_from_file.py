import numpy as np
import matplotlib.pyplot as plt

def read_from_file(filename=""):
    alpha, var, exp = np.loadtxt(fname=filename, skiprows=1, unpack=True)
    return alpha, var, exp

def read_tracker_dt_file(filename=""):
    dt, t_i, t_b, n_i, n_b  = np.loadtxt(fname=filename, skiprows=19, unpack=True,  delimiter="|")
    return dt, t_i, t_b, n_i, n_b


def read_from_file_v2():
    """
    Not in use
    """
    alpha_impor, var_impor, exp_impor = np.loadtxt(fname="generated_data/output_importance.txt", skiprows=1, unpack=True)
    alpha_brute, var_brute, exp_brute = np.loadtxt(fname="generated_data/output_bruteforce.txt", skiprows=1, unpack=True)

    plt.figure()
    plt.plot(alpha_impor, exp_impor, label="importance")
    plt.plot(alpha_brute, exp_brute, label="brute force")
    plt.xlabel("alpha")
    plt.ylabel("energy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #read_from_file()
    #read_from_file_v2()
    read_tracker_dt_file("tracker_dt_importance.txt")
