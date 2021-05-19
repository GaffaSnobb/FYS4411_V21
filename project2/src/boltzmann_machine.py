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

    Returns
    -------
    energy : float
        The local energy.
    """
    energy = 0  # Local energy.
    exponent = exponent_in_wave_function(pos, hidden_biases, weights)
    exponential = np.exp(exponent)
    exponential_negative = np.exp(-exponent)
    # print(f"{exponent=}")
    # print(f"{-exponent=}")
    # print(f"{exponential_negative=}")
    # print(f"{exponential=}")

    for particle in range(N_PARTICLES):
        for dim in range(N_DIMS):
            sum_1 = (weights[particle, dim]/(1 + exponential_negative)).sum()
            sum_2 = (weights[particle, dim]**2*exponential_negative/(1 + exponential_negative)**2).sum()
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

class _RBMVMC:
    """
    Common stuff for both importance sampling and brute force.
    """
    def __init__(self):
        self.learning_rate = 0.01
        
        self.visible_biases = np.random.normal(loc=0.0, scale=0.1, size=(N_PARTICLES, N_DIMS))
        self.hidden_biases = np.random.normal(loc=0.0, scale=0.1, size=N_HIDDEN)
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(N_PARTICLES, N_DIMS, N_HIDDEN))

        self.initial_state()

    def initial_state(self):
        """
        Set all arrays to initial state. Some of the zeroing might be
        superfluous, but better safe than sorry.
        """
        self.acceptance_rate = 0
        self.local_energy_average = 0

        # wave_derivatives_averagee = np.empty(3, dtype=np.ndarray)
        self.wave_derivatives_average = [
            np.zeros_like(self.visible_biases),
            np.zeros_like(self.hidden_biases),
            np.zeros_like(self.weights)
        ]
        self.wave_derivatives_energy_average = [
            np.zeros_like(self.visible_biases),
            np.zeros_like(self.hidden_biases),
            np.zeros_like(self.weights)
        ]
        self.initial_state_addition()
        self.pos_new = np.zeros_like(self.pos_current)
        
        self.wave_current = wave_function(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights
        )

    def solve(self):
        """
        Find the minimum energy using gradient descent.
        """
        self.energies = np.zeros(MAX_ITERATIONS)
        self.times = np.zeros(MAX_ITERATIONS)
        self.acceptance_rates = np.zeros(MAX_ITERATIONS)

        for iteration in range(MAX_ITERATIONS):
            timing = time.time()

            self.initial_state()    # Reset state.
            self.monte_carlo()
            
            self.visible_biases -= self.learning_rate*self.visible_biases_gradient
            self.hidden_biases -= self.learning_rate*self.hidden_biases_gradient
            self.weights -= self.learning_rate*self.weights_gradient 
            self.energies[iteration] = self.local_energy_average
            self.acceptance_rates[iteration] = self.acceptance_rate
            self.times[iteration] = time.time() - timing

            print(f"Energy:          {self.energies[iteration]:.5f} a.u.")
            print(f"Acceptance rate: {self.acceptance_rates[iteration]:.5f}")

        print(f"Average over {MAX_ITERATIONS} iterations: {np.mean(self.energies):.5f} a.u.")
        print(f"Average time per iteration: {np.mean(self.times[1:]):.5f} s")
        print(f"Average acceptance rate:    {np.mean(self.acceptance_rates):.5f}")

class ImportanceSampling(_RBMVMC):
    def __init__(self):
        self.diffusion_coeff = 0.5
        self.time_step = 0.05
        super().__init__()

    def initial_state_addition(self):
        self.pos_current = np.random.normal(loc=0.0, scale=0.001, size=(N_PARTICLES, N_DIMS))
        self.pos_current *= np.sqrt(self.time_step)

        self.qforce_current = quantum_force(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights
        )

    def monte_carlo(self):
        local_energy_partial = local_energy(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights
        )
        wave_derivatives = wave_function_derivative(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights
        )
        for _ in range(N_MC_CYCLES):
            for particle in range(N_PARTICLES):
                """
                Loop over all particles. Move one particle at the time.
                """
                self.pos_new[particle] = self.pos_current[particle]
                self.pos_new[particle] += np.random.normal(loc=0.0, scale=1.0, size=N_DIMS)*np.sqrt(self.time_step)
                self.pos_new[particle] += self.qforce_current[particle]*self.time_step*self.diffusion_coeff
                
                wave_new = wave_function(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights
                )
                qforce_new = quantum_force(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights
                )
                
                greens_function = 0.5*(self.qforce_current[particle] + qforce_new[particle])
                greens_function *= (self.diffusion_coeff*self.time_step*0.5*(self.qforce_current[particle] - qforce_new[particle]) - self.pos_new[particle] + self.pos_current[particle])
                greens_function = np.exp(greens_function.sum())
                
                if np.random.uniform() <= greens_function*(wave_new/self.wave_current)**2:
                    """
                    Metropolis-Hastings.
                    """
                    self.acceptance_rate += 1
                    self.pos_current[particle] = self.pos_new[particle]
                    self.qforce_current[particle] = qforce_new[particle]
                    self.wave_current = wave_new

            local_energy_partial = local_energy(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights
            )
            wave_derivatives = wave_function_derivative(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights
            )
            
            self.wave_derivatives_average[0] += wave_derivatives[0]  # Wrt. visible bias.
            self.wave_derivatives_average[1] += wave_derivatives[1]  # Wrt. hidden bias.
            self.wave_derivatives_average[2] += wave_derivatives[2]  # Wrt. weights.
            
            self.local_energy_average += local_energy_partial

            self.wave_derivatives_energy_average[0] += \
                wave_derivatives[0]*local_energy_partial
            self.wave_derivatives_energy_average[1] += \
                wave_derivatives[1]*local_energy_partial
            self.wave_derivatives_energy_average[2] += \
                wave_derivatives[2]*local_energy_partial

        self.acceptance_rate /= N_MC_CYCLES*N_PARTICLES
        self.local_energy_average /= N_MC_CYCLES
        self.wave_derivatives_energy_average[0] /= N_MC_CYCLES
        self.wave_derivatives_energy_average[1] /= N_MC_CYCLES
        self.wave_derivatives_energy_average[2] /= N_MC_CYCLES
        self.wave_derivatives_average[0] /= N_MC_CYCLES
        self.wave_derivatives_average[1] /= N_MC_CYCLES
        self.wave_derivatives_average[2] /= N_MC_CYCLES
        
        self.visible_biases_gradient = \
            2*(self.wave_derivatives_energy_average[0] - self.wave_derivatives_average[0]*self.local_energy_average)
        self.hidden_biases_gradient = \
            2*(self.wave_derivatives_energy_average[1] - self.wave_derivatives_average[1]*self.local_energy_average)
        self.weights_gradient = \
            2*(self.wave_derivatives_energy_average[2] - self.wave_derivatives_average[2]*self.local_energy_average)

class BruteForce(_RBMVMC):
    def __init__(self):
        self.brute_force_step_size = 0.05
        super().__init__()

    def initial_state_addition(self):
        self.pos_current = np.random.uniform(low=0, high=1, size=(N_PARTICLES, N_DIMS))*self.brute_force_step_size


    def monte_carlo(self):
        for _ in range(N_MC_CYCLES):
            for particle in range(N_PARTICLES):
                """
                Loop over all particles. Move one particle at the time.
                """
                self.pos_new[particle] = self.pos_current[particle]
                self.pos_new[particle] += np.random.uniform(low=0, high=1, size=N_DIMS)*self.brute_force_step_size  # NOTE: Shift by -0.5?
                
                wave_new = wave_function(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights
                )
                
                if np.random.uniform() <= (wave_new/self.wave_current)**2:
                    """
                    Metropolis-Hastings.
                    """
                    self.acceptance_rate += 1
                    self.pos_current[particle] = self.pos_new[particle]
                    self.wave_current = wave_new

            local_energy_partial = local_energy(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights
            )
            wave_derivatives = wave_function_derivative(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights
            )
            
            self.wave_derivatives_average[0] += wave_derivatives[0]  # Wrt. visible bias.
            self.wave_derivatives_average[1] += wave_derivatives[1]  # Wrt. hidden bias.
            self.wave_derivatives_average[2] += wave_derivatives[2]  # Wrt. weights.
            
            self.local_energy_average += local_energy_partial

            self.wave_derivatives_energy_average[0] += \
                wave_derivatives[0]*local_energy_partial
            self.wave_derivatives_energy_average[1] += \
                wave_derivatives[1]*local_energy_partial
            self.wave_derivatives_energy_average[2] += \
                wave_derivatives[2]*local_energy_partial

        self.acceptance_rate /= N_MC_CYCLES*N_PARTICLES
        self.local_energy_average /= N_MC_CYCLES
        self.wave_derivatives_energy_average[0] /= N_MC_CYCLES
        self.wave_derivatives_energy_average[1] /= N_MC_CYCLES
        self.wave_derivatives_energy_average[2] /= N_MC_CYCLES
        self.wave_derivatives_average[0] /= N_MC_CYCLES
        self.wave_derivatives_average[1] /= N_MC_CYCLES
        self.wave_derivatives_average[2] /= N_MC_CYCLES
        
        self.visible_biases_gradient = \
            2*(self.wave_derivatives_energy_average[0] - self.wave_derivatives_average[0]*self.local_energy_average)
        self.hidden_biases_gradient = \
            2*(self.wave_derivatives_energy_average[1] - self.wave_derivatives_average[1]*self.local_energy_average)
        self.weights_gradient = \
            2*(self.wave_derivatives_energy_average[2] - self.wave_derivatives_average[2]*self.local_energy_average)

if __name__ == "__main__":
    np.random.seed(1337)
    N_PARTICLES = 2         # Number of particles.
    N_DIMS = 2              # Number of dimensions.
    N_HIDDEN = 2            # Number of hidden nodes.
    N_MC_CYCLES = int(1e3)  # Number of Monte Carlo cycles.
    INTERACTION = True      # TODO: Double check interaction expression.
    MAX_ITERATIONS = 50

    # I believe this sigma is the std of the normal distribution the visible
    # layers. TODO: Find out how to choose what the std should be.
    SIGMA = 1
    SIGMA_SQUARED = SIGMA**2

    q = ImportanceSampling()
    # q = BruteForce()
    q.solve()