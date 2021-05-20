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

class _RBMVMC:
    """
    Common stuff for both importance sampling and brute force.
    """
    def __init__(
        self,
        n_particles: int,
        n_dims: int,
        n_hidden: int,
        n_mc_cycles: int,
        max_iterations: int,
        learning_rate: float,
        sigma: float,
        interaction: bool
    ) -> None:

        self.learning_rate = learning_rate
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.n_hidden = n_hidden
        self.n_mc_cycles = n_mc_cycles
        self.max_iterations = max_iterations
        self.sigma = sigma
        self.interaction = interaction
        
        self.visible_biases = np.random.normal(loc=0, scale=0.1, size=(self.n_particles, self.n_dims))
        self.hidden_biases = np.random.normal(loc=0, scale=0.1, size=self.n_hidden)
        self.weights = np.random.normal(loc=0, scale=0.1, size=(self.n_particles, self.n_dims, self.n_hidden))

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
            self.weights,
            self.sigma
        )

    def solve(self):
        """
        Find the minimum energy using gradient descent.
        """
        self.energies = np.zeros(self.max_iterations)
        self.times = np.zeros(self.max_iterations)
        self.acceptance_rates = np.zeros(self.max_iterations)

        for iteration in range(self.max_iterations):
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

        print(f"Average over {self.max_iterations} iterations: {np.mean(self.energies):.5f} a.u.")
        print(f"Average time per iteration: {np.mean(self.times[1:]):.5f} s")
        print(f"Average acceptance rate:    {np.mean(self.acceptance_rates):.5f}")

class ImportanceSampling(_RBMVMC):
    def __init__(
        self,
        n_particles: int,
        n_dims: int,
        n_hidden: int,
        n_mc_cycles: int,
        max_iterations: int,
        learning_rate: float,
        sigma: float,
        interaction: bool,
        diffusion_coeff: float,
        time_step: float
    ) -> None:
        
        self.diffusion_coeff = diffusion_coeff
        self.time_step = time_step
        super().__init__(
            n_particles,
            n_dims,
            n_hidden,
            n_mc_cycles,
            max_iterations,
            learning_rate,
            sigma,
            interaction
        )

    def initial_state_addition(self):
        self.pos_current = np.random.normal(loc=0.0, scale=0.001, size=(self.n_particles, self.n_dims))
        self.pos_current *= np.sqrt(self.time_step)

        self.qforce_current = quantum_force(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.sigma
        )

    def monte_carlo(self):
        local_energy_partial = local_energy(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.sigma,
            self.interaction
        )
        wave_derivatives = wave_function_derivative(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.sigma
        )
        for _ in range(self.n_mc_cycles):
            for particle in range(self.n_particles):
                """
                Loop over all particles. Move one particle at the time.
                """
                self.pos_new[particle] = self.pos_current[particle]
                self.pos_new[particle] += np.random.normal(loc=0.0, scale=1.0, size=self.n_dims)*np.sqrt(self.time_step)
                self.pos_new[particle] += self.qforce_current[particle]*self.time_step*self.diffusion_coeff
                
                wave_new = wave_function(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights,
                    self.sigma
                )
                qforce_new = quantum_force(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights,
                    self.sigma
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
                self.weights,
                self.sigma,
                self.interaction
            )
            wave_derivatives = wave_function_derivative(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights,
                self.sigma
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

        self.acceptance_rate /= self.n_mc_cycles*self.n_particles
        self.local_energy_average /= self.n_mc_cycles
        self.wave_derivatives_energy_average[0] /= self.n_mc_cycles
        self.wave_derivatives_energy_average[1] /= self.n_mc_cycles
        self.wave_derivatives_energy_average[2] /= self.n_mc_cycles
        self.wave_derivatives_average[0] /= self.n_mc_cycles
        self.wave_derivatives_average[1] /= self.n_mc_cycles
        self.wave_derivatives_average[2] /= self.n_mc_cycles
        
        self.visible_biases_gradient = \
            2*(self.wave_derivatives_energy_average[0] - self.wave_derivatives_average[0]*self.local_energy_average)
        self.hidden_biases_gradient = \
            2*(self.wave_derivatives_energy_average[1] - self.wave_derivatives_average[1]*self.local_energy_average)
        self.weights_gradient = \
            2*(self.wave_derivatives_energy_average[2] - self.wave_derivatives_average[2]*self.local_energy_average)

class BruteForce(_RBMVMC):
    def __init__(
        self,
        n_particles: int,
        n_dims: int,
        n_hidden: int,
        n_mc_cycles: int,
        max_iterations: int,
        learning_rate: float,
        sigma: float,
        interaction: bool,
        brute_force_step_size: float
    ) -> None:
        
        self.brute_force_step_size = brute_force_step_size
        super().__init__(
            n_particles,
            n_dims,
            n_hidden,
            n_mc_cycles,
            max_iterations,
            learning_rate,
            sigma,
            interaction
        )

    def initial_state_addition(self):
        self.pos_current = np.random.uniform(low=-0.5, high=0.5, size=(self.n_particles, self.n_dims))*self.brute_force_step_size

    def monte_carlo(self):
        for _ in range(self.n_mc_cycles):
            for particle in range(self.n_particles):
                """
                Loop over all particles. Move one particle at the time.
                """
                self.pos_new[particle] = self.pos_current[particle]
                self.pos_new[particle] += np.random.uniform(low=-0.5, high=0.5, size=self.n_dims)*self.brute_force_step_size
                
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

        self.acceptance_rate /= self.n_mc_cycles*self.n_particles
        self.local_energy_average /= self.n_mc_cycles
        self.wave_derivatives_energy_average[0] /= self.n_mc_cycles
        self.wave_derivatives_energy_average[1] /= self.n_mc_cycles
        self.wave_derivatives_energy_average[2] /= self.n_mc_cycles
        self.wave_derivatives_average[0] /= self.n_mc_cycles
        self.wave_derivatives_average[1] /= self.n_mc_cycles
        self.wave_derivatives_average[2] /= self.n_mc_cycles
        
        self.visible_biases_gradient = \
            2*(self.wave_derivatives_energy_average[0] - self.wave_derivatives_average[0]*self.local_energy_average)
        self.hidden_biases_gradient = \
            2*(self.wave_derivatives_energy_average[1] - self.wave_derivatives_average[1]*self.local_energy_average)
        self.weights_gradient = \
            2*(self.wave_derivatives_energy_average[2] - self.wave_derivatives_average[2]*self.local_energy_average)

if __name__ == "__main__":
    np.random.seed(1337)
    N_PARTICLES = 3         # Number of particles.
    N_DIMS = 2              # Number of dimensions.
    N_HIDDEN = 4            # Number of hidden nodes.
    N_MC_CYCLES = int(1e3)  # Number of Monte Carlo cycles.
    INTERACTION = True      # TODO: Double check interaction expression.
    MAX_ITERATIONS = 20
    SIGMA = 1   # Std. of the normal distribution the visible nodes.
    SIGMA_SQUARED = SIGMA**2
    # self.brute_force_step_size = 0.05

    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(1e3),
        max_iterations = 20,
        learning_rate = 0.01,
        sigma = 1,
        interaction = True,
        diffusion_coeff = 0.5,
        time_step = 0.05
    )
    # print(f"{q.hidden_biases.shape}")
    # q = BruteForce(learning_rate)
    q.solve()