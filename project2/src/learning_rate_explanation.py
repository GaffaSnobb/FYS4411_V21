import numpy as np
import matplotlib.pyplot as plt
import other_functions as other
import mpl_rcparams

def main():
    iterations = np.arange(0, 100+1, 1)
    factors = np.array([0.05, 0.1])
    N = len(factors)
    inits = [0.1, 0.5]

    fig, ax = plt.subplots()
    for i in range(len(factors)):
        for j in range(len(inits)):
            init = inits[j]
            factor = factors[i]

            learning_rates = other.variable_learning_rate(
                t = iterations*factor,
                t0 = None,
                t1 = None,
                init = init
            )

            ax.plot(iterations, learning_rates, label=r"$f_{\eta} = $" + f"{factor}, " + r"$\eta_{init} = $" + f"{init}")

    ax.set_xlabel(r"GD iterations ($I$)")
    ax.set_ylabel(r"$\eta$")
    ax.legend()
    fig.savefig(fname="../fig/variable_learning_rate_explanation.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
