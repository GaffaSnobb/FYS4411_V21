"""
Code based off of: http://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/boltzmannmachines.html#representing-the-wave-function.
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
def wave_function(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Trial wave function for the 2-electron system in two dimensions.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: N_PARTICLES x N_DIMS.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: N_PARTICLES x N_DIMS.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: N_HIDDEN.

    weights : numpy.ndarray
        Dimension: N_PARTICLES x N_DIMS x N_HIDDEN
    """
    term_1 = 0
    term_2 = 1
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    
    # term_1 = ((pos - visible_biases)**2).sum()    # This is prob. a replacement for the following two loops.
    for particle in range(N_PARTICLES):
        for dim in range(N_DIMS):
            term_1 += (pos[particle, dim] - visible_biases[particle, dim])**2

    # term_2 = np.product(1 + np.exp(exponent)) # This is prob. a replacement for the following loop.
    for hidden in range(N_HIDDEN):
        term_2 *= (1 + np.exp(exponent[hidden]))
        
    term_1 = np.exp(-term_1/(2*SIGMA_SQUARED))

    return term_1*term_2

@numba.njit
def local_energy(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Analytical local energy for the 2-electron system in two dimensions.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: N_PARTICLES x N_DIMS.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: N_PARTICLES x N_DIMS.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: N_HIDDEN.

    weights : numpy.ndarray
        Dimension: N_PARTICLES x N_DIMS x N_HIDDEN
    """
    energy = 0  # Local energy.
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    exponential = np.exp(exponent)

    for particle in range(N_PARTICLES):
        for dim in range(N_DIMS):
            sum_1 = (weights[particle, dim]/(1 + np.exp(-exponent))).sum()
            sum_2 = (weights[particle, dim]**2*exponential/(1 + exponential)**2).sum()
    
            dlnpsi1 = -(pos[particle, dim] - visible_biases[particle, dim])/SIGMA_SQUARED + sum_1/SIGMA_SQUARED
            dlnpsi2 = -1/SIGMA_SQUARED + sum_2/SIGMA_SQUARED**2
            energy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + pos[particle, dim]**2)
            
    if INTERACTION:
        for particle in range(N_PARTICLES):
            for particle_inner in range(particle):
                distance = ((pos[particle] - pos[particle_inner])**2).sum()
                energy += 1/np.sqrt(distance)
                
    return energy

@numba.njit
def wave_function_derivative(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray
) -> tuple:
    """
    Derivate of wave function as a function of variational parameters.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: N_PARTICLES x N_DIMS.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: N_PARTICLES x N_DIMS.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: N_HIDDEN.

    weights : numpy.ndarray
        Dimension: N_PARTICLES x N_DIMS x N_HIDDEN.

    Returns
    -------
    wave_diff_wrt_visible_bias : numpy.ndarray
        The wave function differentiated with respect to the visible
        bias, divided by the wave function. Dimension: N_PARTICLES x
        N_DIMS.

    wave_diff_wrt_hidden_bias : numpy.ndarray
        The wave function differentiated with respect to the hidden
        bias, divided by the wave function. Dimension: N_HIDDEN.

    wave_diff_wrt_weights : numpy.ndarray
        The wave function differentiated with respect to the weights,
        divided by the wave function. Dimension: N_PARTICLES x N_DIMS x
        N_HIDDEN.
    """    
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    
    wave_diff_wrt_visible_bias = (pos - visible_biases)/SIGMA_SQUARED   # NOTE: This is verified to be correct.
    wave_diff_wrt_hidden_bias = 1/(1 + np.exp(-exponent))   # NOTE: Verify that this is correct.
    wave_diff_wrt_weights = np.zeros_like(weights)
    
    for hidden in range(N_HIDDEN):
        wave_diff_wrt_weights[:, :, hidden] = \
            weights[:, :, hidden]/(SIGMA_SQUARED*(1 + np.exp(-exponent[hidden])))   # NOTE: Verify that this is correct.
            
    return wave_diff_wrt_visible_bias, wave_diff_wrt_hidden_bias, wave_diff_wrt_weights

@numba.njit
def quantum_force(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Quantum force for the two-electron system.

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: N_PARTICLES x N_DIMS.

    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: N_PARTICLES x N_DIMS.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: N_HIDDEN.

    weights : numpy.ndarray
        Dimension: N_PARTICLES x N_DIMS x N_HIDDEN
    """    
    qforce = np.zeros((N_PARTICLES, N_DIMS))
    sum_1 = np.zeros((N_PARTICLES, N_DIMS))
    
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    
    for ih in range(N_HIDDEN):
        sum_1 += weights[:, :, ih]/(1 + np.exp(-exponent[ih]))
    
    qforce = 2*(-(pos - visible_biases)/SIGMA_SQUARED + sum_1/SIGMA_SQUARED)
    
    return qforce

@numba.njit
def exponent_in_wave_function(
    pos: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    The exponent of the exponential factor in the product of the wave
    function.  TODO: A better name for this function. But the current
    name is certainly better than q_fac.

    b_j + sum_i^M (x_i*w_ij/sigma^2).

    Parameters
    ----------
    pos : numpy.ndarray
        Array of particle positions. Dimension: N_PARTICLES x N_DIMS.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: N_HIDDEN.

    weights : numpy.ndarray
        Dimension: N_PARTICLES x N_DIMS x N_HIDDEN.

    Returns
    -------
    exponent : numpy.ndarray
        The exponent of the exponential factor in the product of the
        wave function. Dimension: N_HIDDEN.
    """
    exponent = np.zeros(N_HIDDEN)
    
    for hidden in range(N_HIDDEN):
        exponent[hidden] = (pos*weights[:, :, hidden]).sum()
    
    exponent /= SIGMA_SQUARED
    exponent += hidden_biases
    
    return exponent
    
def energy_minimization(
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray
) -> tuple:
    """
    Compute the local energy and its derivatives.

    Parameters
    ----------
    visible_biases : numpy.ndarray
        The biases of the visible layer. Dimension: N_PARTICLES x N_DIMS.
    
    hidden_biases : numpy.ndarray
        The biases of the hidden nodes. Dimension: N_HIDDEN.

    weights : numpy.ndarray
        Dimension: N_PARTICLES x N_DIMS x N_HIDDEN

    Returns
    -------
    energy : float
    """
    diffusion_coeff = 0.5
    time_step = 0.05
    local_energy_average = 0

    pos_current = np.random.normal(loc=0.0, scale=1.0, size=(N_PARTICLES, N_DIMS))  # NOTE: Aka. visible layer? Yes!
    pos_current *= np.sqrt(time_step)
    pos_new = np.zeros((N_PARTICLES, N_DIMS))
    qforce_current = np.zeros((N_PARTICLES, N_DIMS))
    qforce_new = np.zeros((N_PARTICLES, N_DIMS))

    # wave_derivatives_averagee = np.empty(3, dtype=np.ndarray)
    wave_derivatives_average = [np.zeros_like(visible_biases), np.zeros_like(hidden_biases), np.zeros_like(weights)]
    wave_derivatives_energy_average = [np.zeros_like(visible_biases), np.zeros_like(hidden_biases), np.zeros_like(weights)]
    
    wave_current = wave_function(pos_current, visible_biases, hidden_biases, weights)
    qforce_current = quantum_force(pos_current, visible_biases, hidden_biases, weights)

    for _ in range(N_MC_CYCLES):
        for particle in range(N_PARTICLES):
            """
            Loop over all particles. Move one particle at the time.
            """
            pos_new[particle] = pos_current[particle]
            pos_new[particle] += np.random.normal(loc=0.0, scale=1.0, size=N_DIMS)*np.sqrt(time_step)
            pos_new[particle] += qforce_current[particle]*time_step*diffusion_coeff
            
            wave_new = wave_function(pos_new, visible_biases, hidden_biases, weights)
            qforce_new = quantum_force(pos_new, visible_biases, hidden_biases, weights)
            
            greens_function = 0.5*(qforce_current[particle] + qforce_new[particle])
            greens_function *= (diffusion_coeff*time_step*0.5*(qforce_current[particle] - qforce_new[particle]) - pos_new[particle] + pos_current[particle])
            greens_function = np.exp(greens_function.sum())
            
            if np.random.uniform() <= greens_function*(wave_new/wave_current)**2:
                """
                Metropolis-Hastings.
                """
                pos_current[particle] = pos_new[particle]
                qforce_current[particle] = qforce_new[particle]
                wave_current = wave_new

        local_energy_partial = local_energy(
            pos_current,
            visible_biases,
            hidden_biases,
            weights
        )
        wave_derivatives = wave_function_derivative(
            pos_current,
            visible_biases,
            hidden_biases,
            weights
        )
        
        wave_derivatives_average[0] += wave_derivatives[0]  # Wrt. visible bias.
        wave_derivatives_average[1] += wave_derivatives[1]  # Wrt. hidden bias.
        wave_derivatives_average[2] += wave_derivatives[2]  # Wrt. weights.
        
        local_energy_average += local_energy_partial

        wave_derivatives_energy_average[0] += wave_derivatives[0]*local_energy_partial
        wave_derivatives_energy_average[1] += wave_derivatives[1]*local_energy_partial
        wave_derivatives_energy_average[2] += wave_derivatives[2]*local_energy_partial
    
    local_energy_average /= N_MC_CYCLES
    wave_derivatives_energy_average[0] /= N_MC_CYCLES
    wave_derivatives_energy_average[1] /= N_MC_CYCLES
    wave_derivatives_energy_average[2] /= N_MC_CYCLES
    wave_derivatives_average[0] /= N_MC_CYCLES
    wave_derivatives_average[1] /= N_MC_CYCLES
    wave_derivatives_average[2] /= N_MC_CYCLES
    gradient = []
    gradient.append(2*(wave_derivatives_energy_average[0] - wave_derivatives_average[0]*local_energy_average))
    gradient.append(2*(wave_derivatives_energy_average[1] - wave_derivatives_average[1]*local_energy_average))
    gradient.append(2*(wave_derivatives_energy_average[2] - wave_derivatives_average[2]*local_energy_average))
    
    return local_energy_average, gradient


np.random.seed(1337)
N_PARTICLES = 2         # Number of particles.
N_DIMS = 2              # Number of dimensions.
N_HIDDEN = 2            # Number of hidden nodes.
N_MC_CYCLES = 10_000    # Number of Monte Carlo cycles.
INTERACTION = False     # TODO: Implement interaction.

# I believe this sigma is the std of the normal distribution the visible
# layers. TODO: Find out how to choose what the std should be.
SIGMA = 1
SIGMA_SQUARED = SIGMA**2

visible_biases = np.random.normal(loc=0.0, scale=0.001, size=(N_PARTICLES, N_DIMS))
hidden_biases = np.random.normal(loc=0.0, scale=0.001, size=N_HIDDEN)
weights = np.random.normal(loc=0.0, scale=0.001, size=(N_PARTICLES, N_DIMS, N_HIDDEN))

energy = 0
learning_rate = 0.001
max_iterations = 5
energies = np.zeros(max_iterations)

for iteration in range(max_iterations):
    timing = time.time()

    energy, gradient = energy_minimization(visible_biases, hidden_biases, weights)
    visible_biases_gradient = gradient[0]
    hidden_biases_gradient = gradient[1]
    weights_gradient = gradient[2]
    
    visible_biases -= learning_rate*visible_biases_gradient
    hidden_biases -= learning_rate*hidden_biases_gradient
    weights -= learning_rate*weights_gradient 
    energies[iteration] = energy

    print("Energy:", energy)
    print(f"Time: {time.time() - timing}")