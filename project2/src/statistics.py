import numpy as np
import matplotlib.pyplot as plt

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

        # estimate variance of x
        s[i] = np.var(x)

        # estimate the error
        error_array[i] = (s[i]/n)**.5


    # generate the test observator M_k from the theorem
    M = (np.cumsum( ((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
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



def main():
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()

    learning_rates = [0.1, 0.02, 0.04, 0.06, 0.08]
    colors = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:pink"]

    for ic, lr in enumerate(learning_rates):
        fname = f"../out/importance_True_{lr}.npy"
        data = np.load(fname)

        iterations = data.shape[0]
        mc_cycles = data.shape[1]

        avg  = np.zeros(iterations)
        orig = np.zeros(iterations)
        best = np.zeros(iterations)
        iter = np.zeros(iterations)

        for i in range(iterations):
            iter[i] = i
            x = data[i]

            avg[i], best[i], orig[i], k, error_array = block(x, verbose=False)

        ax0.errorbar(iter, avg, yerr=best, color = colors[ic], label=f"learning rate: {lr}")

        ax1.plot(iter, best, color = colors[ic], label=f"blocking error, {lr}")
        ax1.plot(iter, orig, color = colors[ic], linestyle = "dashed", label=f"original error, {lr}")

    ax0.set_xlabel("Iterations")
    ax0.set_ylabel("Energy")

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Error")

    ax0.legend()
    ax1.legend()

    plt.show()

if __name__ == '__main__':
    main()
