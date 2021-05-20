"""
Code based off of: http://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/boltzmannmachines.html#representing-the-wave-function.
Cheat sheet:
x: visible layer
a: visible bias
h: hidden layer
b: hidden bias
W: interaction weights
"""
from typing import Union, Type
import sys, time
import numpy as np
import other_functions as other

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
        
        self.initial_state()
        self.reset_state()

    def initial_state(
        self,
        loc_scale_all: Union[tuple, Type[None]] = (0, 0.1),
        loc_scale_visible_biases: Union[tuple, Type[None]] = None,
        loc_scale_hidden_biases: Union[tuple, Type[None]] = None,
        loc_scale_weights: Union[tuple, Type[None]] = None,
    ):
        """
        Set the initial state of all nodes, weights and biases.

        Parameters
        ----------
        loc_scale_all:
            The loc (mean) and scale (std) of the normal distribution
            of all initial distributions.

        loc_scale_visible_biases:
            The loc (mean) and scale (std) of the normal distribution
            of the visible biases.

        loc_scale_hidden_biases:
            The loc (mean) and scale (std) of the normal distribution
            of the hidden biases.

        loc_scale_weights:
            The loc (mean) and scale (std) of the normal distribution
            of the weights.
        """
        if (loc_scale_all is None) and (
            (loc_scale_visible_biases is None) or
            (loc_scale_hidden_biases is None) or
            (loc_scale_weights is None)
        ):
            msg = f"loc_scale_all cant be None when"
            msg += f" either of the other inputs are None."
            raise ValueError(msg)

        if loc_scale_visible_biases is None:
            loc_visible_biases = loc_scale_all[0]
            scale_visible_biases = loc_scale_all[1]

        if loc_scale_hidden_biases is None:
            loc_hidden_biases = loc_scale_all[0]
            scale_hidden_biases = loc_scale_all[1]

        if loc_scale_weights is None:
            loc_weights = loc_scale_all[0]
            scale_weights = loc_scale_all[1]

        self.visible_biases = np.random.normal(
            loc = loc_visible_biases,
            scale = scale_visible_biases,
            size = (self.n_particles, self.n_dims)
        )
        self.hidden_biases = np.random.normal(
            loc = loc_hidden_biases,
            scale = scale_hidden_biases,
            size = self.n_hidden
        )
        self.weights = np.random.normal(
            loc = loc_weights,
            scale = scale_weights,
            size = (self.n_particles, self.n_dims, self.n_hidden)
        )

    def reset_state(self):
        """
        Set all arrays to initial state. Some of the zeroing might be
        superfluous, but better safe than sorry. Nodes, weights and
        biases are not reset.
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
        self.reset_state_addition()
        self.pos_new = np.zeros_like(self.pos_current)
        
        self.wave_current = other.wave_function(
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

            self.reset_state()
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

    def reset_state_addition(self):
        self.pos_current = np.random.normal(loc=0.0, scale=0.001, size=(self.n_particles, self.n_dims))
        self.pos_current *= np.sqrt(self.time_step)

        self.qforce_current = other.quantum_force(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.sigma
        )

    def monte_carlo(self):
        local_energy_partial = other.local_energy(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.sigma,
            self.interaction
        )
        wave_derivatives = other.wave_function_derivative(
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
                
                wave_new = other.wave_function(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights,
                    self.sigma
                )
                qforce_new = other.quantum_force(
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

            local_energy_partial = other.local_energy(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights,
                self.sigma,
                self.interaction
            )
            wave_derivatives = other.wave_function_derivative(
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

    def reset_state_addition(self):
        self.pos_current = np.random.uniform(low=-0.5, high=0.5, size=(self.n_particles, self.n_dims))*self.brute_force_step_size

    def monte_carlo(self):
        for _ in range(self.n_mc_cycles):
            for particle in range(self.n_particles):
                """
                Loop over all particles. Move one particle at the time.
                """
                self.pos_new[particle] = self.pos_current[particle]
                self.pos_new[particle] += np.random.uniform(low=-0.5, high=0.5, size=self.n_dims)*self.brute_force_step_size
                
                wave_new = other.wave_function(
                    self.pos_new,
                    self.visible_biases,
                    self.hidden_biases,
                    self.weights,
                    self.sigma
                )
                
                if np.random.uniform() <= (wave_new/self.wave_current)**2:
                    """
                    Metropolis-Hastings.
                    """
                    self.acceptance_rate += 1
                    self.pos_current[particle] = self.pos_new[particle]
                    self.wave_current = wave_new

            local_energy_partial = other.local_energy(
                self.pos_current,
                self.visible_biases,
                self.hidden_biases,
                self.weights,
                self.sigma,
                self.interaction
            )
            wave_derivatives = other.wave_function_derivative(
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

if __name__ == "__main__":
    np.random.seed(1337)
    # self.brute_force_step_size = 0.05

    # q = ImportanceSampling(
    #     n_particles = 2,
    #     n_dims = 2,
    #     n_hidden = 2,
    #     n_mc_cycles = int(1e3),
    #     max_iterations = 20,
    #     learning_rate = 0.01,
    #     sigma = 1,              # Std. of the normal distribution the visible nodes.
    #     interaction = True,
    #     diffusion_coeff = 0.5,
    #     time_step = 0.05
    # )
    q = BruteForce(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(1e5),
        max_iterations = 20,
        learning_rate = 0.01,
        sigma = 1,              # Std. of the normal distribution the visible nodes.
        interaction = True,
        brute_force_step_size = 0.05
    )
    q.solve()