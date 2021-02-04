import numpy as np
import matplotlib.pyplot as plt

def read_from_file():
    alpha, var, exp = np.loadtxt(fname="generated_data/output.txt", skiprows=1, unpack=True)
    plt.plot(alpha, exp)
    plt.xlabel("alpha")
    plt.ylabel("energy")
    plt.show()




if __name__ == "__main__":
    read_from_file()