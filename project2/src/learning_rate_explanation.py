import numpy as np
import matplotlib.pyplot as plt
import other_functions as other
import mpl_rcparams

def main():
    """
    Produce a plot to show how the variable learning rate evolves.
    """
    iterations = np.arange(0, 50+1, 1)
    factors = np.array([0.02, 0.06, 0.1])
    N = len(factors)

    fig, ax = plt.subplots()
    for i in range(N):
        learning_rates = other.variable_learning_rate(
            t = iterations*factors[i],
            t0 = None,
            t1 = None,
            init = 0.18
        )
        ax.plot(
            iterations,
            learning_rates,
            label = r"$f_{\eta} = $" + f"{factors[i]}, " + r"$\eta_{init} = $" + f"{0.18}"
        )
    
    ax.set_xlabel(r"GD iterations ($I$)")
    ax.set_ylabel(r"$\eta$")
    ax.legend()
    fig.savefig(fname="../fig/variable_learning_rate_explanation.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()