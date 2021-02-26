import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import norm

from read_from_file import read_energy_from_file, read_from_file

# Returns mean of bootstrap samples

def stat(data):
    return np.mean(data)

# Bootstrap algorithm
def bootstrap(data, statistic, R):
    t = np.zeros(R);
    n = len(data);
    inds = np.arange(n);
    t0 = time()

    # non-parametric bootstrap
    for i in range(R):
        t[i] = statistic(data[np.random.randint(0, n, n)])

    # analysis
    print(f"Runtime: {time()-t0:.4f} sec")
    print("Bootstrap Statistics : ")
    print(f"data: {np.mean(data):.3f}, std.: {np.std(data):.3f}, bias: {np.mean(t):.3f} std. error: {np.std(t):.3f}")
    print()
    return t


print(45*"_", "\n")
print("Importance")
print(45*"_", "\n")


file = "generated_data/output_energy_importance.txt"
alpha, energy = read_energy_from_file(file)

for i in range(len(alpha)):
    print(f"alpha: {alpha[i]:.2f}")
    t = bootstrap(energy[:,i], stat, len(energy[:,i]))
    #n, binsboot, patches = plt.hist(t, 50, facecolor='red', alpha=0.75)
    #y = norm.pdf(binsboot, np.mean(t), np.std(t))
    #plt.plot(binsboot, y, 'k--', linewidth=1)
    #plt.show()


quit()
mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*np.random.randn(datapoints)

# bootstrap returns the data sample

t = bootstrap(x, stat, datapoints)

# the histogram of the bootstrapped  data
n, binsboot, patches = plt.hist(t, 50, facecolor='red', alpha=0.75)

# add a 'best fit' line
y = norm.pdf(binsboot, np.mean(t), np.std(t))
plt.plot(binsboot, y, 'r--', linewidth=1)
plt.xlabel('Smarts')
plt.ylabel('Probability')

plt.show()


void VMC::write_to_file(std::string fpath)
{
    outfile.open(fpath, std::ios::out);
    outfile << std::setw(20) << "alpha";
    outfile << std::setw(20) << "variance_energy";
    outfile << std::setw(21) << "expected_energy";
    outfile << std::setw(21) << "n_particles\n";

    for (int i = 0; i < n_variations; i++)
    {   /*
        Write data to file.
        */
        outfile << std::setw(20) << std::setprecision(10);
        outfile << alphas(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_variances(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_expectations(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << n_particles << "\n";
    }
    outfile.close();
}
