import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_all_files
from time import time

def block(x, verbose=True):
    """
    Credit: Marius Jonsson
    Jonsson, M. (2018). Standard error estimation by an automated blocking method. Physical Review E, 98(4), 043304.
    """
    # preliminaries
    n = len(x)
    d = int(np.log2(n))
    s, gamma, error_array = np.zeros(d), np.zeros(d), np.zeros(d)
    mu = np.mean(x)
    t0 = time()

    # Calculate the autocovariance and variance for the data
    gamma[0] = (n)**(-1)*np.sum( (x[0:(n-1)] - mu) * (x[1:n] - mu) )
    s[0] = np.var(x)
    error_array[0] = (s[0]/n)**.5

    # estimate the auto-covariance and variances for each blocking transformation
    for i in np.arange(1, d):

        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

        n = len(x)

        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum( (x[0:(n-1)] - mu) * (x[1:n] - mu) )

        #gam = 0
        #for j in range(n-1):
        #    gam += (x[j] - mu)*(x[j+1]- mu)

        #gamma[i] = (n)**(-1) * gam

        # estimate variance of x
        s[i] = np.var(x)

        # estimate the error
        error_array[i] = (s[i]/n)**.5


    # generate the test observator M_k from the theorem
    M = (np.cumsum( ((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    # alpha=0.01
    """
    q = np.array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894,
                18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967,
                27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306,
                36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820,
                44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])
    """
    # alpha= 0.05
    q = np.array([3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507,
                 16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296,
                 27.587, 28.869, 30.144, 31.410, 32.671, 33.924, 35.172, 36.415,
                 37.652, 38.885, 40.113, 41.337, 42.557, 43.773, 44.985, 46.194,
                 47.400, 48.602, 49.802, 50.998, 52.192, 53.384, 54.572, 55.758,
                 56.942, 58.124, 59.304, 60.481, 61.656, 62.830, 64.001, 65.171,
                 66.339, 67.505, 68.669, 69.832, 70.993, 72.153, 73.311, 74.468,
                 75.624, 76.778, 77.931, 79.082, 80.232, 81.381, 82.529, 83.675,
                 84.821, 85.965, 87.108, 88.250, 89.391, 90.531, 91.670, 92.808,
                 93.945, 95.081, 96.217, 97.351, 98.484, 99.617, 100.749, 101.879,
                 103.010, 104.139, 105.267, 106.395, 107.522, 108.648, 109.773,
                 110.898, 112.022, 113.145, 114.268, 115.390, 116.511, 117.632,
                 118.752, 119.871, 120.990, 122.108, 123.225, 124.342, 124.342])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0, d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        if verbose:
            print ("Warning: Use more data")

    best_error = error_array[k]
    original_error = error_array[0]

    if verbose:
        print(f"avg: {mu:.6f}, error(orig): {original_error:.6f}, error(block): {best_error:.6f}, iterations: {k}\n")

    return mu, best_error, original_error, k, error_array



def mean_var_error(data):
    mean = np.mean(data)
    var = np.var(data)
    error = (var/len(data))**.5
    return mean, var, error



def analyze(f_energy, type=""):
    """
    print the results of blocking for all alphas.
    input
    ----------------------------
    f_energy: str, filename
    type:     str, when running for Gradient descent the columns with zero
                    alpha is removed (the ones that are not filled)

    """
    print(45*"_"+"\n", type, 45*"_"+"\n", sep="\n")

    alphas, energies, n_particles = read_energy_from_file(f_energy)
    if type == "GradientDescent":
        alphas, energies, n_particles = read_energy_from_file(f_energy, clip=True)

    for alpha in range(len(alphas)):
        print(f"alpha: {alphas[alpha]:.2f}")

        data = energies[:,alpha]

        best_error, error = block(data)
        iter = np.arange(len(error))
        best = np.ones(len(error)) * best_error



def plot_error(f_energy, type="", alpha_value = None):
    """
    plot the error as a function of number of iterations

    input
    ----------------------------
        f_energy:    str, filename of data
        type:        str, todo: unprof solution maybe fix later
        alpha_value: int, where the wanted alpha value is, is no value is
                            specified, this defaults to the middle alpha.

    """
    alphas, energies, n_particles = read_energy_from_file(f_energy)
    if type == "GradientDescent":
        alphas, energies, n_particles = read_energy_from_file(f_energy, clip=True)

    if alpha_value:
        alpha = alpha_value
    else:
        alpha = int(len(alphas)/2)

    print(45*"_"+"\n", type, 45*"_"+"\n", sep="\n")
    print(f"alpha: {alphas[alpha]:.2f}")

    data = energies[:,alpha]
    best_error, error = block(data)

    iter = np.arange(len(error))
    best = np.ones(len(error)) * best_error

    plt.figure()
    plt.grid()
    plt.title(type)
    plt.plot(iter, error, color="k", label=r"$Error, \alpha=$"+f"{alphas[alpha]}")
    plt.plot(iter, best, linestyle="dashed", color="tab:red", label="Optimal")

    plt.xticks(np.arange(min(iter), max(iter)+1, 1.0))
    plt.xlabel("Blocking iterations, k")
    plt.ylabel(r"Sample Error, $\sqrt{\sigma^2_k \ / \ n_k}$")
    plt.legend()
    plt.show()


def blocking_analysis(n_particles, n_dims, mc_cycles, method, numerical=False, interaction=False):

    input = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = n_dims,
        filter_n_mc_cycles = mc_cycles,
        filter_step_size = None,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = "energies"
    )

    # Sort elements based on the number of particles.
    input.sort(key=lambda elem: elem.n_particles)

    # Get the array of alpha values
    alphas = input[0].data[0,:]

    # new filename
    fname_blocking = f"blocking_{input[0].fname}"
    file = open("generated_data/" + fname_blocking, "w")

    header = "alpha\tenergy\t\toriginal_error\tblocking_error\titerations"
    print(header)

    file.write(header+"\n")

    for i in range(len(alphas)):
        # Loop over alpha values and do blocking to find error

        data = input[0].data[1:,i]
        energy, blocking_error, original_error, iterations, error_array = block(data, verbose=False)

        s = f"{alphas[i]:.1f}\t{energy:.6f}\t{original_error:.6f}\t{blocking_error:.6f}\t{iterations}"
        print(s)

        file.write(s+"\n")

        iter = np.arange(len(error_array))
        block_line = np.ones(len(error_array)) * blocking_error
        orig_line =  np.ones(len(error_array)) * original_error

        plt.figure()
        plt.grid()
        plt.plot(iter, error_array, ".", color="k", label=r"$Error, \alpha=$"+f"{alphas[i]}")
        plt.plot(iter, block_line, linestyle="dashed", color="tab:red", label="Optimal")
        plt.plot(iter, orig_line, linestyle="dashed", color="tab:blue", label="Original")

        plt.xticks(np.arange(min(iter), max(iter)+1, 1.0))
        plt.xlabel("Blocking iterations, k")
        plt.ylabel(r"Sample Error, $\sqrt{\sigma^2_k \ / \ n_k}$")
        plt.legend()
        plt.show()


#    energy, best_error, original, iterations, error = block(input[0].data[1:,0], verbose=False)
#    iter = np.arange(len(error))
#    best = np.ones(len(error)) * best_error
#
#    plt.figure()
#    plt.grid()
#    plt.plot(iter, error, ".", color="k", label=r"$Error, \alpha=$"+f"{alphas[1]}")
#    plt.plot(iter, best, linestyle="dashed", color="tab:red", label="Optimal")
#
#    plt.xticks(np.arange(min(iter), max(iter)+1, 1.0))
#    plt.xlabel("Blocking iterations, k")
#    plt.ylabel(r"Sample Error, $\sqrt{\sigma^2_k \ / \ n_k}$")
#    plt.legend()
#    plt.show()



def main():
    print(45*"_"+"\n", "Brute-Force", 45*"_"+"\n", sep="\n")
    blocking_analysis(method = "brute",
                      n_particles = 10,
                      n_dims = 3,
                      mc_cycles= int(2**20),
                      numerical=False,
                      interaction=False)

#    print(45*"_"+"\n", "Importance", 45*"_"+"\n", sep="\n")
#    blocking_analysis(method = "importance",
#                      n_particles = 10,
#                      n_dims = 3,
#                      mc_cycles= int(2**20),
#                      numerical=False,
#                      interaction=False)


if __name__ == '__main__':
    main()

    """
    brute_3d = read_all_files(
        filter_method = "brute",
        filter_n_particles = None,
        filter_n_dims = 3,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )

    variance = brute_3d[0].data[:, 1]
    energy = brute_3d[0].data[:,2]
    n_particles = brute_3d[0].n_particles
    n_mc_cycles = brute_3d[0].n_mc_cycles
    alphas = brute_3d[0].data[:, 0]

    error = (variance/n_mc_cycles)**0.5

    for i in range(len(alphas)):
        print(f"alpha: {alphas[i]:.2f}")
        print(f"avg: {energy[i]:.6f}, variance: {variance[i]:.6f}, error: {error[i]:.6f}\n")
    """
