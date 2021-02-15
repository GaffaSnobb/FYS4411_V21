import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_from_file, read_tracker_dt_file

def brute_and_importance(fname_brute, fname_impor):
    alpha_brute, var_brute, exp_brute = read_from_file(fname_brute)
    alpha_impor, var_impor, exp_impor = read_from_file(fname_impor)

    fig = plt.figure()
    plt.plot(alpha_impor, exp_impor, color="k", label="importance")
    plt.plot(alpha_brute, exp_brute, color="tab:blue", label="brute force")
    plt.xlabel("alpha")
    plt.ylabel("energy")
    plt.legend()
    fig.savefig("../fig/compare_brute_importance.png")
    plt.show()



def importance_time_step(path):
    dt, t_i, t_b, n_i, n_b  = read_tracker_dt_file("tracker_dt_importance.txt")

    fig = plt.figure()
    for i in range(len(dt)):
        filename = f"{path}/output_importance_{dt[i]}.txt"
        alpha, var, exp = read_from_file(filename)
        plt.plot(alpha, exp, label=f"time_step = {dt[i]}")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"$\langle E \rangle $")
    plt.legend()
    plt.show()

    """
    fig = plt.figure()
    plt.plot(dt, n_i, label="accepted importance")
    plt.plot(dt, n_b, label="accepted brute force")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("number")
    plt.legend()
    plt.show()
    """

if __name__ == "__main__":
    path = "generated_data"
    fname_bruteforce = f"{path}/output_bruteforce.txt"
    fname_importance = f"{path}/output_importance_0.4.txt"
    brute_and_importance(fname_brute=fname_bruteforce, fname_impor=fname_importance)

    #importance_time_step(path)
