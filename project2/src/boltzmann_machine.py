"""
2-electron VMC code for 2dim quantum dot with importance sampling
Using gaussian rng for new positions and Metropolis- Hastings 
Added restricted boltzmann machine method for dealing with the wavefunction
RBM code based heavily off of:
https://github.com/CompPhysics/ComputationalPhysics2/tree/gh-pages/doc/Programs/BoltzmannMachines/MLcpp/src/CppCode/ob

Cheat sheet:
x: visible layer
a: visible bias
h: hidden layer
b: hidden bias
W: interaction weights
"""

import sys, time
import numpy as np
import numba

@numba.njit
def wave_function(pos, visible_biases, hidden_biases, weights):
    """
    Trial wave function for the 2-electron quantum dot in two dims.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: n_particles x n_dims.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: n_particles x n_dims.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: n_hidden.

    weights : numpy.ndarray
        Dimension: n_particles x n_dims x n_hidden
    """
    psi_1 = 0.0
    psi_2 = 1.0
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    
    for particle in range(n_particles):
        for dim in range(n_dims):
            psi_1 += (pos[particle, dim] - visible_biases[particle, dim])**2
            
    for hidden in range(n_hidden):
        psi_2 *= (1.0 + np.exp(exponent[hidden]))
        
    psi_1 = np.exp(-psi_1/(2*sigma**2))

    return psi_1*psi_2

@numba.njit
def local_energy(pos, visible_biases, hidden_biases, weights):
    """
    Local energy  for the 2-electron quantum dot in two dims, using
    analytical local energy.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: n_particles x n_dims.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: n_particles x n_dims.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: n_hidden.

    weights : numpy.ndarray
        Dimension: n_particles x n_dims x n_hidden
    """
    energy = 0                  # Local energy.
    
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)

    for particle in range(n_particles):
        for dim in range(n_dims):
            sum_1 = (weights[particle, dim]/(1 + np.exp(-exponent))).sum()
            Q_exp = np.exp(exponent)
            sum_2 = (weights[particle, dim]**2*Q_exp/(1 + Q_exp)**2).sum()
    
            dlnpsi1 = -(pos[particle, dim] - visible_biases[particle, dim]) /sigma_squared + sum_1/sigma_squared
            dlnpsi2 = -1/sigma_squared + sum_2/sigma_squared**2
            energy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + pos[particle, dim]**2)
            
    if interaction:
        for iq1 in range(n_particles):
            for iq2 in range(iq1):
                distance = ((pos[iq1] - pos[iq2])**2).sum()
                energy += 1/np.sqrt(distance)
                
    return energy

# @numba.njit
def wave_function_derivative(pos, visible_biases, hidden_biases, weights):
    """
    Derivate of wave function as a function of variational parameters.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: n_particles x n_dims.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: n_particles x n_dims.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: n_hidden.

    weights : numpy.ndarray
        Dimension: n_particles x n_dims x n_hidden
    """    
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    
    WfDer = [None, None, np.zeros_like(weights)]
    
    WfDer[0] = (pos - visible_biases)/sigma_squared
    WfDer[1] = 1/(1 + np.exp(-exponent))
    
    for hidden in range(n_hidden):
        WfDer[2][:, :, hidden] = \
            weights[:, :, hidden]/(sigma_squared*(1 + np.exp(-exponent[hidden])))
            
    return  WfDer

def quantum_force(pos, visible_biases, hidden_biases, weights):
    """
    Setting up the quantum force for the two-electron quantum dot.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: n_particles x n_dims.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: n_particles x n_dims.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: n_hidden.

    weights : numpy.ndarray
        Dimension: n_particles x n_dims x n_hidden
    """
    sigma = 1.0
    sigma_squared = sigma**2
    
    qforce = np.zeros((n_particles, n_dims))
    sum_1 = np.zeros((n_particles, n_dims))
    
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    
    for ih in range(n_hidden):
        sum_1 += weights[:, :, ih]/(1 + np.exp(-exponent[ih]))
    
    qforce = 2*(-(pos - visible_biases)/sigma_squared + sum_1/sigma_squared)
    
    return qforce

@numba.njit
def exponent_in_wave_function(pos, hidden_biases, weights):
    """
    The exponent of the exponential factor in the product of the wave
    function.

    b_j + sum_i^M (x_i*w_ij/sigma^2).

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: n_particles x n_dims.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: n_hidden.

    weights : numpy.ndarray
        Dimension: n_particles x n_dims x n_hidden.

    Returns
    -------
    Q : numpy.ndarray
        The exponent of the exponential factor in the product of the
        wave function. Dimension: n_hidden.
    """
    exponent = np.zeros(n_hidden)
    
    for hidden in range(n_hidden):
        exponent[hidden] = (pos*weights[:, :, hidden]).sum()
    
    exponent /= sigma_squared
    exponent += hidden_biases
    
    return exponent
    
def energy_minimization(visible_biases, hidden_biases, weights):
    """
    Computing the local energy and the derivative.

    Parameters
    ----------
    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: n_particles x n_dims.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: n_hidden.

    weights : numpy.ndarray
        Dimension: n_particles x n_dims x n_hidden

    Returns
    -------
    energy : float
    """
    n_mc_cycles = 10_000    # Number of Monte Carlo cycles.
    
    # Parameters in the Fokker-Planck simulation of the quantum force.
    diffusion_coeff = 0.5
    time_step = 0.05

    pos_current = np.random.normal(loc=0.0, scale=1.0, size=(n_particles, n_dims))  # NOTE: Aka. visible layer?
    pos_current *= np.sqrt(time_step)
    pos_new = np.zeros((n_particles, n_dims))
    qforce_current = np.zeros((n_particles, n_dims))
    qforce_new = np.zeros((n_particles, n_dims))

    energy = 0

    DeltaPsi = [np.zeros_like(visible_biases), np.zeros_like(hidden_biases), np.zeros_like(weights)]
    DerivativePsiE = [np.zeros_like(visible_biases), np.zeros_like(hidden_biases), np.zeros_like(weights)]
    
    wave_current = wave_function(pos_current, visible_biases, hidden_biases, weights)
    qforce_current = quantum_force(pos_current, visible_biases, hidden_biases, weights)

    for _ in range(n_mc_cycles):
        """
        Trial position moving one particle at the time.
        """
        for i in range(n_particles):
            """
            Loop over all particles.
            """
            pos_new[i] = pos_current[i] + np.random.normal(loc=0.0, scale=1.0, size=n_dims)*np.sqrt(time_step) + qforce_current[i]*time_step*diffusion_coeff
            wave_new = wave_function(pos_new, visible_biases, hidden_biases, weights)
            qforce_new = quantum_force(pos_new, visible_biases, hidden_biases, weights)
            
            greens_function = 0.5*(qforce_current[i] + qforce_new[i])*(diffusion_coeff*time_step*0.5*(qforce_current[i] - qforce_new[i]) - pos_new[i] + pos_current[i])
            greens_function = np.exp(greens_function.sum())
            
            if np.random.uniform() <= greens_function*(wave_new/wave_current)**2:
                """
                Metropolis-Hastings.
                """
                pos_current[i] = pos_new[i]
                qforce_current[i] = qforce_new[i]

                wave_current = wave_new

        de = local_energy(pos_current, visible_biases, hidden_biases, weights)
        DerPsi = wave_function_derivative(pos_current, visible_biases, hidden_biases, weights)
        
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
n_dims = 2          # Number of dimensions.
n_hidden = 2        # Number of hidden nodes.

sigma = 1.0
sigma_squared = sigma**2

interaction = False

# guess for parameters
visible_biases = np.random.normal(loc=0.0, scale=0.001, size=(n_particles, n_dims))
hidden_biases = np.random.normal(loc=0.0, scale=0.001, size=n_hidden)
weights = np.random.normal(loc=0.0, scale=0.001, size=(n_particles, n_dims, n_hidden))
# Set up iteration using stochastic gradient method
energy = 0
# Learning rate eta, max iterations, need to change to adaptive learning rate
eta = 0.001
max_iterations = 5
energies = np.zeros(max_iterations)

for iteration in range(max_iterations):
    timing = time.time()
    energy, EDerivative = energy_minimization(visible_biases, hidden_biases, weights)
    agradient = EDerivative[0]
    bgradient = EDerivative[1]
    wgradient = EDerivative[2]
    visible_biases -= eta*agradient
    hidden_biases -= eta*bgradient
    weights -= eta*wgradient 
    energies[iteration] = energy

    print("Energy:", energy)
    print(f"Time: {time.time() - timing}")