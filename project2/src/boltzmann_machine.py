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
import sys, time, os
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
        interaction: bool,
        omega: float
    ) -> None:

        self.learning_rate = learning_rate
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.n_hidden = n_hidden
        self.n_mc_cycles = n_mc_cycles
        self.max_iterations = max_iterations
        self.sigma = sigma
        self.interaction = interaction
        self.omega = omega

        self.initial_state()
        self.reset_state()

        self.call_solve = False

    def initial_state(
        self,
        loc_scale_all: Union[tuple, Type[None]] = (0, 0.1),
        loc_scale_visible_biases: Union[tuple, Type[None]] = None,
        loc_scale_hidden_biases: Union[tuple, Type[None]] = None,
        loc_scale_weights: Union[tuple, Type[None]] = None,
    ) -> None:
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
        self.loc_scale_all = loc_scale_all
        self.loc_scale_hidden_biases = loc_scale_hidden_biases
        self.loc_scale_visible_biases = loc_scale_visible_biases
        self.loc_scale_weights = loc_scale_weights

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
        else:
            loc_visible_biases = loc_scale_visible_biases[0]
            scale_visible_biases = loc_scale_visible_biases[1]

        if loc_scale_hidden_biases is None:
            loc_hidden_biases = loc_scale_all[0]
            scale_hidden_biases = loc_scale_all[1]
        else:
            loc_hidden_biases = loc_scale_hidden_biases[0]
            scale_hidden_biases = loc_scale_hidden_biases[1]

        if loc_scale_weights is None:
            loc_weights = loc_scale_all[0]
            scale_weights = loc_scale_all[1]
        else:
            loc_weights = loc_scale_weights[0]
            scale_weights = loc_scale_weights[1]

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

        self._generate_paths()

    def reset_state(self) -> None:
        """
        Set all arrays to initial state. Some of the zeroing might be
        superfluous, but better safe than sorry. Nodes, weights and
        biases are not reset.
        """
        self.acceptance_rate = 0
        self.local_energy_average = 0

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

        self.energy_mc = np.zeros(self.n_mc_cycles)

    def _generate_paths(self) -> None:
        """
        Generate all needed file and directory names and paths.
        """
        self.parent_data_directory = "generated_data"
        
        self.current_data_directory = f"{self.prefix}_"
        self.current_data_directory += f"{self.n_particles}_"
        self.current_data_directory += f"{self.n_dims}_"
        self.current_data_directory += f"{self.n_hidden}_"
        self.current_data_directory += f"{self.n_dims}_"
        self.current_data_directory += f"{self.n_mc_cycles}_"
        self.current_data_directory += f"{self.max_iterations}_"
        self.current_data_directory += f"{self.learning_rate}_"
        self.current_data_directory += f"{self.sigma}_"
        self.current_data_directory += f"{self.interaction}_"
        self.current_data_directory += f"{self.omega}_"

        if self.loc_scale_all is not None:
            self.current_data_directory += f"all{self.loc_scale_all}_".replace(" ", "")
        if self.loc_scale_visible_biases is not None:
            self.current_data_directory += f"a{self.loc_scale_visible_biases}_".replace(" ", "")
        if self.loc_scale_hidden_biases is not None:
            self.current_data_directory += f"b{self.loc_scale_hidden_biases}_".replace(" ", "")
        if self.loc_scale_weights is not None:
            self.current_data_directory += f"w{self.loc_scale_weights}_".replace(" ", "")
        
        self.current_data_directory += f"{self.postfix}"
        self.full_data_path = f"{self.parent_data_directory}/{self.current_data_directory}"

    def solve(self, verbose: bool = True) -> None:
        """
        Find the minimum energy using gradient descent.
        """
        self.call_solve = True
        if os.path.isdir(f"{self.full_data_path}"):
            self._load_state()
            return

        self.energies = np.zeros(self.max_iterations)
        self.times = np.zeros(self.max_iterations)
        self.acceptance_rates = np.zeros(self.max_iterations)
        self.energy_mc_iter = np.zeros((self.max_iterations, self.n_mc_cycles))

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

            self.energy_mc_iter[iteration, :] = self.energy_mc

            if verbose:
                print(f"Energy:          {self.energies[iteration]:.5f} a.u.")
                print(f"Acceptance rate: {self.acceptance_rates[iteration]:.5f}")

        if verbose:
            print(f"Average over {self.max_iterations} iterations: {np.mean(self.energies):.5f} a.u.")
            print(f"Average time per iteration: {np.mean(self.times[1:]):.5f} s")
            print(f"Average acceptance rate:    {np.mean(self.acceptance_rates):.5f}")

        self._save_state()

    def _save_state(self) -> None:
        """
        Save relevant data as numpy arrays.
        """
        if not self.call_solve:
            print(f"Cannot save state before running 'solve'. Exiting...")
            sys.exit(0)

        if not os.path.isdir(self.parent_data_directory):
            os.mkdir(self.parent_data_directory)
        
        if not os.path.isdir(self.full_data_path):
            os.mkdir(self.full_data_path)

        np.save(f"{self.full_data_path}/energy_mc_iter.npy", self.energy_mc_iter)
        np.save(f"{self.full_data_path}/acceptance_rates.npy", self.acceptance_rates)
        np.save(f"{self.full_data_path}/energies.npy", self.energies)

    def _load_state(self) -> None:
        """
        Load relevant data as numpy arrays.
        """
        if not self.call_solve:
            print(f"Cannot load state before running 'solve'. Exiting...")
            sys.exit(0)
        self.energy_mc_iter = np.load(f"{self.full_data_path}/energy_mc_iter.npy")
        self.acceptance_rates = np.load(f"{self.full_data_path}/acceptance_rates.npy")
        self.energies = np.load(f"{self.full_data_path}/energies.npy")

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
        omega: float,
        diffusion_coeff: float,
        time_step: float
    ) -> None:

        self.diffusion_coeff = diffusion_coeff
        self.time_step = time_step
        self.prefix = "importance"
        self.postfix = f"{self.diffusion_coeff}_{self.time_step}"
        super().__init__(
            n_particles,
            n_dims,
            n_hidden,
            n_mc_cycles,
            max_iterations,
            learning_rate,
            sigma,
            interaction,
            omega
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
        # These two calls can prob. be removed.
        # local_energy_partial = other.local_energy(
        #     self.pos_current,
        #     self.visible_biases,
        #     self.hidden_biases,
        #     self.weights,
        #     self.sigma,
        #     self.interaction,
        #     self.omega
        # )
        # wave_derivatives = other.wave_function_derivative(
        #     self.pos_current,
        #     self.visible_biases,
        #     self.hidden_biases,
        #     self.weights,
        #     self.sigma
        # )

        for cycle in range(self.n_mc_cycles):
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
                self.interaction,
                self.omega
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

            self.energy_mc[cycle] = local_energy_partial


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
        omega: float,
        brute_force_step_size: float
    ) -> None:

        self.brute_force_step_size = brute_force_step_size
        self.prefix = "brute"
        self.postfix = f"{self.brute_force_step_size}"
        super().__init__(
            n_particles,
            n_dims,
            n_hidden,
            n_mc_cycles,
            max_iterations,
            learning_rate,
            sigma,
            interaction,
            omega
        )

    def reset_state_addition(self) -> None:
        self.pos_current = np.random.uniform(low=-0.5, high=0.5, size=(self.n_particles, self.n_dims))*self.brute_force_step_size

    def monte_carlo(self) -> None:
        for cycle in range(self.n_mc_cycles):
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
                self.interaction,
                self.omega
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

            self.energy_mc[cycle] = local_energy_partial

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
    omega = 1/4
    sigma = np.sqrt(1/omega)
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 3,
        n_hidden = 2,
        n_mc_cycles = int(2**12),
        max_iterations = 20,
        learning_rate = 0.01,
        sigma = sigma,              # Std. of the normal distribution the visible nodes.
        interaction = True,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05
    )
    # q = BruteForce(
    #     n_particles = 2,
    #     n_dims = 2,
    #     n_hidden = 2,
    #     n_mc_cycles = int(2**12),
    #     max_iterations = 20,
    #     learning_rate = 0.01,
    #     sigma = 1,              # Std. of the normal distribution the visible nodes.
    #     interaction = True,
    #     brute_force_step_size = 0.05
    # )
    q.initial_state(
        loc_scale_hidden_biases = (0, 0.1),
        loc_scale_visible_biases = (0, 0.1),
        loc_scale_weights = (0, 0.1)
    )
    q.solve()
