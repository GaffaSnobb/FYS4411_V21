import numba
import numpy as np

@numba.njit()
def wave_function(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray,
    sigma: float
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
    exponent = exponent_in_wave_function(pos, hidden_biases, weights, sigma)
    n_particles, n_dims = pos.shape
    n_hidden = hidden_biases.shape[0]
    
    # term_1 = ((pos - visible_biases)**2).sum()    # This is prob. a replacement for the following two loops.
    for particle in range(n_particles):
        for dim in range(n_dims):
            term_1 += (pos[particle, dim] - visible_biases[particle, dim])**2

    # term_2 = np.product(1 + np.exp(exponent)) # This is prob. a replacement for the following loop.
    for hidden in range(n_hidden):
        term_2 *= (1 + np.exp(exponent[hidden]))
        
    term_1 = np.exp(-term_1/(2*sigma**2))

    return term_1*term_2

@numba.njit
def local_energy(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    interaction: bool
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

    Returns
    -------
    energy : float
        The local energy.
    """
    energy = 0  # Local energy.
    exponent = exponent_in_wave_function(pos, hidden_biases, weights, sigma)
    exponential_negative = np.exp(-exponent)
    n_particles, n_dims = pos.shape
    sigma_squared = sigma**2

    for particle in range(n_particles):
        for dim in range(n_dims):
            sum_1 = (weights[particle, dim]/(1 + exponential_negative)).sum()
            sum_2 = (weights[particle, dim]**2*exponential_negative/(1 + exponential_negative)**2).sum()
            dlnpsi1 = -(pos[particle, dim] - visible_biases[particle, dim])/sigma_squared + sum_1/sigma_squared
            dlnpsi2 = -1/sigma_squared + sum_2/sigma_squared**2
            energy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + pos[particle, dim]**2)

    if interaction:
        for particle in range(n_particles):
            for particle_inner in range(particle):
                distance = ((pos[particle] - pos[particle_inner])**2).sum()
                energy += 1/np.sqrt(distance)
                
    return energy

@numba.njit
def wave_function_derivative(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray,
    sigma: float
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
    exponent = exponent_in_wave_function(pos, hidden_biases, weights, sigma)
    n_hidden = hidden_biases.shape[0]
    sigma_squared = sigma**2
    
    wave_diff_wrt_visible_bias = (pos - visible_biases)/sigma_squared   # NOTE: This is verified to be correct.
    wave_diff_wrt_hidden_bias = 1/(1 + np.exp(-exponent))   # NOTE: This is verified to be correct.
    wave_diff_wrt_weights = np.zeros_like(weights)
    
    for hidden in range(n_hidden):
        wave_diff_wrt_weights[:, :, hidden] = \
            weights[:, :, hidden]/(sigma_squared*(1 + np.exp(-exponent[hidden])))   # NOTE: Verify that this is correct. Should 'weights' actually be 'pos'?
            
    return wave_diff_wrt_visible_bias, wave_diff_wrt_hidden_bias, wave_diff_wrt_weights

@numba.njit
def quantum_force(
    pos: np.ndarray,
    visible_biases: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray,
    sigma: float
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
    n_particles, n_dims = pos.shape
    n_hidden = hidden_biases.shape[0]
    sigma_squared = sigma**2

    qforce = np.zeros((n_particles, n_dims))
    sum_1 = np.zeros((n_particles, n_dims))
    
    exponent = exponent_in_wave_function(pos, hidden_biases, weights, sigma)
    
    for ih in range(n_hidden):
        sum_1 += weights[:, :, ih]/(1 + np.exp(-exponent[ih]))
    
    qforce = 2*(-(pos - visible_biases)/sigma_squared + sum_1/sigma_squared)

    return qforce

@numba.njit
def exponent_in_wave_function(
    pos: np.ndarray,
    hidden_biases: np.ndarray,
    weights: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    The exponent of the exponential factor in the product of the wave
    function.

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
    n_hidden = hidden_biases.shape[0]
    exponent = np.zeros(n_hidden)
    
    for hidden in range(n_hidden):
        exponent[hidden] = (pos*weights[:, :, hidden]).sum()
    
    exponent /= sigma**2
    exponent += hidden_biases
    
    return exponent
