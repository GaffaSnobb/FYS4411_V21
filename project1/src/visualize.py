import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_from_file, read_tracker_dt_file

def brute_and_importance(fname_brute, fname_importance):
    alpha_brute, var_brute, exp_brute = read_from_file(fname_brute)
    alpha_importance, var_importance, exp_importance = read_from_file(fname_importance)

    fig = plt.figure()
    plt.plot(alpha_importance, exp_importance, color="k", label="importance")
    plt.plot(alpha_brute, exp_brute, color="tab:blue", label="brute force")
    #plt.xlabel("alpha")
    #plt.ylabel("energy")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"$\langle E \rangle $")
    plt.legend()
    fig.savefig("../fig/compare_brute_importance.png")
    plt.show()

def gradient_descent(fname):
    alpha, var, exp = read_from_file(fname)
    plt.plot(alpha, exp, ".")
    plt.show()


def plot_all(fname_brute, fname_importance, fname_gradient_descent):
    """
    temporary plot function
    """
    alpha_brute, var_brute, exp_brute = read_from_file(fname_brute)
    alpha_impor, var_impor, exp_impor = read_from_file(fname_importance)
    alpha_gradi, var_gradi, exp_gradi = read_from_file(fname_gradient_descent)

    fig = plt.figure()
    plt.plot(alpha_impor, exp_impor, color="k", label="importance")
    plt.plot(alpha_brute, exp_brute, color="tab:blue", label="brute force")
    plt.plot(alpha_gradi, exp_gradi, ".", color="tab:green", label = "gradient descent")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"$\langle E \rangle $")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = "generated_data"
    fname_brute_force = f"{path}/output_brute_force.txt"
    fname_importance = f"{path}/output_importance.txt"
    fname_gradient_descent = f"{path}/output_gradient_descent.txt"
    brute_and_importance(fname_brute=fname_brute_force, fname_importance=fname_importance)
    gradient_descent(fname_gradient_descent)

    #plot_all(fname_brute_force, fname_importance, fname_gradient_descent)
