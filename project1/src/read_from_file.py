import numpy as np
import matplotlib.pyplot as plt

def read_from_file():
    alpha, var, exp = np.loadtxt(fname="outfile.txt", skiprows=1, unpack=True)
    plt.plot(alpha, exp)
    plt.show()




if __name__ == "__main__":
    read_from_file()