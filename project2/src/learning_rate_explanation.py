import numpy as np
import matplotlib.pyplot as plt
import other_functions as other

def main():
    t = np.linspace(0, 50, 100)
    t0 = np.arange(0, 6+1, 1)
    t1 = np.arange(30, 100+5, 5)

    for i in range(len(t1)):
        # learning_rates = other.variable_learning_rate(t, t0[i], t1=75)
        learning_rates = other.variable_learning_rate(t, t0=3, t1=t1[i])
        plt.plot(t, learning_rates, label=f"{t1[i]=}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()