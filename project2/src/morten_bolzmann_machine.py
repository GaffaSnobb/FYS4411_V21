# 2-electron VMC code for 2dim quantum dot with importance sampling
# Using gaussian rng for new positions and Metropolis- Hastings 
# Added restricted boltzmann machine method for dealing with the wavefunction
# RBM code based heavily off of:
# https://github.com/CompPhysics/ComputationalPhysics2/tree/gh-pages/doc/Programs/BoltzmannMachines/MLcpp/src/CppCode/ob
import sys, time, timeit
import numpy as np

def wave_function(r, a, biases, weights):
    """
    Trial wave function for the 2-electron quantum dot in two dims.
    """
    sigma = 1.0
    psi_1 = 0.0
    psi_2 = 1.0
    Q = q_fac(r, biases, weights)
    
    for iq in range(n_particles):
        for ix in range(n_dims):
            psi_1 += (r[iq,ix] - a[iq,ix])**2
            
    for ih in range(n_hidden):
        psi_2 *= (1.0 + np.exp(Q[ih]))
        
    psi_1 = np.exp(-psi_1/(2*sigma**2))

    return psi_1*psi_2

def local_energy(r, a, biases, weights):
    """
    Local energy  for the 2-electron quantum dot in two dims, using
    analytical local energy.
    """
    sigma =  1.0
    sigma_squared = sigma**2
    energy = 0                  # Local energy.
    
    Q = q_fac(r, biases, weights)

    for iq in range(n_particles):
        for ix in range(n_dims):
            sum_1 = (weights[iq, ix]/(1 + np.exp(-Q))).sum()
            Q_exp = np.exp(Q)
            sum_2 = (weights[iq, ix]**2*Q_exp/(1 + Q_exp)**2).sum()
    
            dlnpsi1 = -(r[iq, ix] - a[iq, ix]) /sigma_squared + sum_1/sigma_squared
            dlnpsi2 = -1/sigma_squared + sum_2/sigma_squared**2
            energy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq, ix]**2)
            
    if interaction:
        for iq1 in range(n_particles):
            for iq2 in range(iq1):
                distance = ((r[iq1] - r[iq2])**2).sum()
                energy += 1/np.sqrt(distance)
                
    return energy

# Derivate of wave function ansatz as function of variational parameters
def DerivativeWFansatz(r,a,biases,weights):
    
    sigma = 1.0
    sigma_squared = sigma**2
    
    Q = q_fac(r, biases, weights)
    
    WfDer = [None, None, np.zeros_like(weights)]
    
    WfDer[0] = (r - a)/sigma_squared
    WfDer[1] = 1 / (1 + np.exp(-Q))
    
    for ih in range(n_hidden):
        WfDer[2][:, :, ih] = weights[:, :, ih] / (sigma_squared*(1 + np.exp(-Q[ih])))
            
    return  WfDer

# Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
def quantum_force(r, a, biases, weights):

    sigma = 1.0
    sigma_squared = sigma**2
    
    qforce = np.zeros((n_particles, n_dims), np.double)
    sum_1 = np.zeros((n_particles, n_dims), np.double)
    
    Q = q_fac(r,biases,weights)
    
    for ih in range(n_hidden):
        sum_1 += weights[:,:,ih]/(1 + np.exp(-Q[ih]))
    
    qforce = 2*(-(r - a)/sigma_squared + sum_1/sigma_squared)
    
    return qforce
    
def q_fac(r, biases, weights):
    Q = np.zeros(n_hidden, np.double)
    
    for ih in range(n_hidden):
        Q[ih] = (r*weights[:, :, ih]).sum()
        
    Q = biases + Q
    
    return Q
    
# Computing the derivative of the energy and the energy 
def energy_minimization(a, biases, weights):

    n_mc_cycles = 10_000    # Number of Monte Carlo cycles.
    
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5     # NOTE: What is this?
    time_step = 0.05

    pos_current = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, n_dims))
    pos_current *= np.sqrt(time_step)
    pos_new = np.zeros((n_particles, n_dims), np.double)
    qforce_current = np.zeros((n_particles, n_dims), np.double)
    qforce_new = np.zeros((n_particles, n_dims), np.double)

    energy = 0.0
    de = 0.0

    DeltaPsi = [np.zeros_like(a), np.zeros_like(biases), np.zeros_like(weights)]
    DerivativePsiE = [np.zeros_like(a), np.zeros_like(biases), np.zeros_like(weights)]
    
    wave_current = wave_function(pos_current, a, biases, weights)
    qforce_current = quantum_force(pos_current, a, biases, weights)

    for _ in range(n_mc_cycles):
        """
        Trial position moving one particle at the time.
        """
        for i in range(n_particles):
            """
            Loop over all particles.
            """
            pos_new[i] = pos_current[i] + np.random.normal(loc=0.0, scale=1.0, size=n_dims)*np.sqrt(time_step) + qforce_current[i]*time_step*D
            wave_new = wave_function(pos_new, a, biases, weights)
            qforce_new = quantum_force(pos_new, a, biases, weights)
            
            greens_function = 0.5*(qforce_current[i] + qforce_new[i])*(D*time_step*0.5*(qforce_current[i] - qforce_new[i]) - pos_new[i] + pos_current[i])
            greens_function = greens_function.sum()
            greens_function = np.exp(greens_function)
            
            if np.random.uniform() <= greens_function*wave_new**2/wave_current**2:
                """
                Metropolis-Hastings.
                """
                pos_current[i] = pos_new[i]
                qforce_current[i] = qforce_new[i]

                wave_current = wave_new

        de = local_energy(pos_current, a, biases, weights)
        DerPsi = DerivativeWFansatz(pos_current, a, biases, weights)
        
        DeltaPsi[0] += DerPsi[0]
        DeltaPsi[1] += DerPsi[1]
        DeltaPsi[2] += DerPsi[2]
        
        energy += de

        DerivativePsiE[0] += DerPsi[0]*de
        DerivativePsiE[1] += DerPsi[1]*de
        DerivativePsiE[2] += DerPsi[2]*de
    
    # Averaging:
    energy /= n_mc_cycles
    DerivativePsiE[0] /= n_mc_cycles
    DerivativePsiE[1] /= n_mc_cycles
    DerivativePsiE[2] /= n_mc_cycles
    DeltaPsi[0] /= n_mc_cycles
    DeltaPsi[1] /= n_mc_cycles
    DeltaPsi[2] /= n_mc_cycles
    EnergyDer = []
    EnergyDer.append(2*(DerivativePsiE[0] - DeltaPsi[0]*energy))
    EnergyDer.append(2*(DerivativePsiE[1] - DeltaPsi[1]*energy))
    EnergyDer.append(2*(DerivativePsiE[2] - DeltaPsi[2]*energy))
    
    return energy, EnergyDer


np.random.seed(1337)
n_particles = 2     # Number of particles.
n_dims = 2           # Number of dimensions.
n_hidden = 2        # NOTE: Find out if this is the number of hidden nodes.

interaction = False

# guess for parameters
a = np.random.normal(loc=0.0, scale=0.001, size=(n_particles, n_dims))
biases = np.random.normal(loc=0.0, scale=0.001, size=n_hidden)
weights = np.random.normal(loc=0.0, scale=0.001, size=(n_particles, n_dims, n_hidden))
# Set up iteration using stochastic gradient method
energy = 0
# Learning rate eta, max iterations, need to change to adaptive learning rate
eta = 0.001
max_iterations = 5
np.seterr(invalid='raise')  # NOTE: Unsure why this is here.
energies = np.zeros(max_iterations)

for _ in range(max_iterations):
    timing = time.time()
    energy, EDerivative = energy_minimization(a, biases, weights)
    agradient = EDerivative[0]
    bgradient = EDerivative[1]
    wgradient = EDerivative[2]
    a -= eta*agradient
    biases -= eta*bgradient 
    weights -= eta*wgradient 
    energies[iter] = energy

    print("Energy:", energy)
    print(f"Time: {time.time() - timing}")