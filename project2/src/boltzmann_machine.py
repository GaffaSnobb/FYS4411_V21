# Code based off of: http://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/boltzmannmachines.html#representing-the-wave-function.
from typing import Union, Type
import sys, time, os
import numpy as np
import numba
import other_functions as other

class Blocking:
    @staticmethod
    @numba.njit
    def blocking(x: np.ndarray) -> tuple:
        """
        Credit: Marius Jonsson
        Jonsson, M. (2018). Standard error estimation by an automated
        blocking method. Physical Review E, 98(4), 043304.
        """
        # preliminaries
        n = len(x)
        d = int(np.log2(n))
        s, gamma, error_array = np.zeros(d), np.zeros(d), np.zeros(d)
        mu = np.mean(x)

        # Calculate the autocovariance and variance for the data
        gamma[0] = (n)**(-1)*np.sum( (x[0:(n-1)] - mu) * (x[1:n] - mu) )
        s[0] = np.var(x)
        error_array[0] = (s[0]/n)**.5

        # estimate the auto-covariance and variances for each blocking transformation
        for i in np.arange(1, d):

            # perform blocking transformation
            x = 0.5*(x[0::2] + x[1::2])

            n = len(x)

            # estimate autocovariance of x
            gamma[i] = (n)**(-1)*np.sum( (x[0:(n-1)] - mu) * (x[1:n] - mu) )

            # estimate variance of x
            s[i] = np.var(x)

            # estimate the error
            error_array[i] = (s[i]/n)**.5


        # generate the test observator M_k from the theorem
        M = (np.cumsum( ((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1] )  )[::-1]

        # we need a list of magic numbers
        # alpha= 0.05
        q = np.array([3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507,
                    16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296,
                    27.587, 28.869, 30.144, 31.410, 32.671, 33.924, 35.172, 36.415,
                    37.652, 38.885, 40.113, 41.337, 42.557, 43.773, 44.985, 46.194,
                    47.400, 48.602, 49.802, 50.998, 52.192, 53.384, 54.572, 55.758,
                    56.942, 58.124, 59.304, 60.481, 61.656, 62.830, 64.001, 65.171,
                    66.339, 67.505, 68.669, 69.832, 70.993, 72.153, 73.311, 74.468,
                    75.624, 76.778, 77.931, 79.082, 80.232, 81.381, 82.529, 83.675,
                    84.821, 85.965, 87.108, 88.250, 89.391, 90.531, 91.670, 92.808,
                    93.945, 95.081, 96.217, 97.351, 98.484, 99.617, 100.749, 101.879,
                    103.010, 104.139, 105.267, 106.395, 107.522, 108.648, 109.773,
                    110.898, 112.022, 113.145, 114.268, 115.390, 116.511, 117.632,
                    118.752, 119.871, 120.990, 122.108, 123.225, 124.342, 124.342])

        # use magic to determine when we should have stopped blocking
        for k in np.arange(0, d):
            if(M[k] < q[k]):
                break

        best_error = error_array[k]
        original_error = error_array[0]

        return mu, best_error, original_error, k, error_array

class _RBMVMC(Blocking):
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
        learning_rate_input: Union[float, str, list, dict],
        sigma: float,
        interaction: bool,
        omega: float,
        parent_data_directory: Union[None, str] = None,
        rng_seed: Union[int, None] = None
    ) -> None:
        """
        n_particles:
            The number of particles. Allowed values are 1 and 2.

        n_dims:
            The number of spatial dimensions. Allowed values are 1, 2,
            and 3.

        n_hidden:
            The number of hidden nodes.

        n_mc_cycles:
            The number of Monte Carlo cycles. Must be a power of 2 to be
            blocking compatible.

        max_iterations:
            The number of gradient descent iterations.

        learning_rate_input:
            The learning rate. Constant learning rate if input is float
            or int. Variable learning rate if input is a dict on the
            form {"init": x, "factor":, y} or
            {"t0": x, "t1": y, "factor": z}.

        sigma:
            The standard deviation of the Gaussian part of the Gaussian-
            binary. Usually set to sigma^2 = 1/omega for better
            resemblance of the ground state wave function.

        interaction:
            Toggle the interaction term in the local energy on / off.

        omega:
            The harmonic oscillator potential frequency.

        parent_data_directory:
            Name of data directory.

        rng_seed:
            Seed for the random number generator.
        """
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.learning_rate = learning_rate_input
        self.learning_rate_input = learning_rate_input
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.n_hidden = n_hidden
        self.n_mc_cycles = n_mc_cycles
        self.max_iterations = max_iterations
        self.sigma = sigma
        self.interaction = interaction
        self.omega = omega
        self.parent_data_directory = parent_data_directory

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

        Raises
        ------
        ValueError:
            If loc_scale_all is None and any of the other inputs are
            None.
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

        self.visible_biases = self.rng.normal(
            loc = loc_visible_biases,
            scale = scale_visible_biases,
            size = (self.n_particles, self.n_dims)
        )
        self.hidden_biases = self.rng.normal(
            loc = loc_hidden_biases,
            scale = scale_hidden_biases,
            size = self.n_hidden
        )
        self.weights = self.rng.normal(
            loc = loc_weights,
            scale = scale_weights,
            size = (self.n_particles, self.n_dims, self.n_hidden)
        )
        self.blocking_all = np.zeros(self.max_iterations)
        self._generate_paths()

    def reset_state(self) -> None:
        """
        Set all arrays to initial state. Some of the zeroing might be
        superfluous, but better safe than sorry. Nodes, weights and
        biases are not reset.
        """
        self.acceptance_rate = 0
        self.local_energy_average = 0

        self.wave_derivatives_average = np.empty(3, dtype=np.ndarray)
        self.wave_derivative_average_wrt_visible_bias = np.zeros_like(self.visible_biases)
        self.wave_derivative_average_wrt_hidden_bias = np.zeros_like(self.hidden_biases)
        self.wave_derivative_average_wrt_weights = np.zeros_like(self.weights)
        
        self.wave_derivatives_energy_average = np.empty(3, dtype=np.ndarray)
        self.wave_derivative_energy_average_wrt_visible_bias = np.zeros_like(self.visible_biases)
        self.wave_derivative_energy_average_wrt_hidden_bias = np.zeros_like(self.hidden_biases)
        self.wave_derivative_energy_average_wrt_weights = np.zeros_like(self.weights)

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
        self.main_data_directory = "tmp"
            
        self.current_data_directory = f"{self.prefix}_"
        self.current_data_directory += f"{self.n_particles}_"
        self.current_data_directory += f"{self.n_dims}_"
        self.current_data_directory += f"{self.n_hidden}_"
        self.current_data_directory += f"{self.n_dims}_"
        self.current_data_directory += f"{self.n_mc_cycles}_"
        self.current_data_directory += f"{self.max_iterations}_"
        
        if isinstance(self.learning_rate_input, dict):
            self.current_data_directory += \
                f"{tuple(self.learning_rate_input.values())}_".replace(" ", "")
        else:
            self.current_data_directory += f"{self.learning_rate_input}_"
        
        self.current_data_directory += f"{self.sigma}_"
        self.current_data_directory += f"{self.interaction}_"
        self.current_data_directory += f"{self.omega}_"

        if self.loc_scale_all is not None:
            self.current_data_directory += \
                f"all{self.loc_scale_all}_".replace(" ", "")
        if self.loc_scale_visible_biases is not None:
            self.current_data_directory += \
                f"a{self.loc_scale_visible_biases}_".replace(" ", "")
        if self.loc_scale_hidden_biases is not None:
            self.current_data_directory += \
                f"b{self.loc_scale_hidden_biases}_".replace(" ", "")
        if self.loc_scale_weights is not None:
            self.current_data_directory += \
                f"w{self.loc_scale_weights}_".replace(" ", "")
        if self.rng_seed is not None:
            self.current_data_directory += f"{self.rng_seed}_"
        
        self.current_data_directory += f"{self.postfix}"
        if self.parent_data_directory is not None:
            self.full_data_path = \
                f"{self.main_data_directory}/{self.parent_data_directory}/{self.current_data_directory}"
        else:
            self.full_data_path = \
                f"{self.main_data_directory}/{self.current_data_directory}"

    def solve(
        self,
        verbose: bool = True,
        save_state: bool = True,
        load_state: bool = True,
        calculate_blocking_all: bool = True
    ) -> None:
        """
        Find the minimum energy using gradient descent.

        Parameters
        ----------
        verbose:
            Toggle energy, acceptance rate and learning rate print on /
            off.
        
        save_state:
            Toggle save state on / off.

        load_state:
            Toggle load state on / off. If False, save_state is also set
            to False.

        calculate_blocking_all:
            Toggle calculate blocking for all iterations on / off.
            Reasonable to turn this off if only uncertainties for the
            last iteration is needed and the number of Monte Carlo
            cycles is large.
        """
        self.calculate_blocking_all = calculate_blocking_all
        self.call_solve = True

        if not load_state:
            save_state = False
            print(f"Load state is {load_state}, setting save state to {save_state}")

        if os.path.isdir(f"{self.full_data_path}") and load_state:
            self._load_state()
            return

        self.energies = np.zeros(self.max_iterations)
        self.times = np.zeros(self.max_iterations)
        self.acceptance_rates = np.zeros(self.max_iterations)
        self.energy_mc_iter = np.zeros((self.max_iterations, self.n_mc_cycles))

        for iteration in range(self.max_iterations):
            """
            Perform the gradient descent.
            """
            timing = time.time()

            if self.learning_rate_input == "variable":
                """
                Variable learning rate with default t0 and t1 parameter
                values.
                """
                self.learning_rate = other.variable_learning_rate(
                    t = 1*iteration,
                    t0 = 2.5,
                    t1 = 50,
                    init = None
                )
            
            elif isinstance(self.learning_rate_input, dict):
                """
                Variable learning rate with input t0 and t1 parameter
                values.
                """
                if "init" not in self.learning_rate_input.keys():
                    t = self.learning_rate_input["factor"]*iteration
                    t0 = self.learning_rate_input["t0"]
                    t1 = self.learning_rate_input["t1"]
                    self.learning_rate = other.variable_learning_rate(
                        t = t,
                        t0 = t0,
                        t1 = t1,
                        init = None
                    )
                else:
                    """
                    Variable learning rate with factor and initial
                    learning rate input.
                    """
                    t = self.learning_rate_input["factor"]*iteration
                    init = self.learning_rate_input["init"]
                    self.learning_rate = other.variable_learning_rate(
                        t = t,
                        t0 = None,
                        t1 = None,
                        init = init
                    )

            self.reset_state()
            self.acceptance_rate = np.zeros(1)  # Hack to get numba to work
            self.local_energy_average = np.zeros(1)
            self.monte_carlo()

            self.visible_biases -= self.learning_rate*self.visible_biases_gradient
            self.hidden_biases -= self.learning_rate*self.hidden_biases_gradient
            self.weights -= self.learning_rate*self.weights_gradient
            self.energies[iteration] = self.local_energy_average
            self.acceptance_rates[iteration] = self.acceptance_rate
            self.times[iteration] = time.time() - timing

            self.energy_mc_iter[iteration, :] = self.energy_mc

            if self.calculate_blocking_all:
                _, self.blocking_all[iteration], _, _, _ = \
                    self.blocking(self.energy_mc_iter[iteration, :])

            if verbose:
                print(f"Energy:          {self.energies[iteration]:.5f} a.u.")
                print(f"Acceptance rate: {self.acceptance_rates[iteration]:.5f}")
                print(f"Learning rate:   {self.learning_rate}")
                print(f"Iteration {iteration + 1} of {self.max_iterations}")

        if verbose:
            print(f"Average over {self.max_iterations} iterations: {np.mean(self.energies):.5f} a.u.")
            print(f"Average time per iteration: {np.mean(self.times[1:]):.5f} s")
            print(f"Average acceptance rate:    {np.mean(self.acceptance_rates):.5f}")

        if self.calculate_blocking_all:
            """
            No need to calculate blocking for the final iteration if
            blocking for all iterations is already calculated.
            """
            self.blocking_final = self.blocking_all[-1]
        else:
            """
            Calculate blocking for only the final iteration.
            """
            _, self.blocking_final, _, _, _ = self.blocking(self.energy_mc_iter[-1, :])
        self.blocking_final = np.array([self.blocking_final])

        if save_state: self._save_state()

        return

    def _save_state(self) -> None:
        """
        Save relevant data as numpy arrays.
        """
        if not self.call_solve:
            print(f"Cannot save state before running 'solve'. Exiting...")
            sys.exit(0)

        if not os.path.isdir(self.main_data_directory):
            """
            main_data_directory/
            """
            os.mkdir(self.main_data_directory)

        if not os.path.isdir(f"{self.main_data_directory}/{self.parent_data_directory}"):
            """
            main_data_directory/parent_data_directory/
            """
            if self.parent_data_directory is not None:
                os.mkdir(f"{self.main_data_directory}/{self.parent_data_directory}")
        
        if not os.path.isdir(self.full_data_path):
            """
            main_data_directory/parent_data_directory/current_data_directory/
            """
            os.mkdir(self.full_data_path)

        np.save(f"{self.full_data_path}/energy_mc_iter.npy", self.energy_mc_iter)
        np.save(f"{self.full_data_path}/acceptance_rates.npy", self.acceptance_rates)
        np.save(f"{self.full_data_path}/energies.npy", self.energies)
        np.save(f"{self.full_data_path}/times.npy", self.times)
        np.save(f"{self.full_data_path}/blocking_final.npy", self.blocking_final)
        if self.calculate_blocking_all:
            np.save(f"{self.full_data_path}/blocking_all.npy", self.blocking_all)

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
        self.times = np.load(f"{self.full_data_path}/times.npy")
        self.blocking_final = np.load(f"{self.full_data_path}/blocking_final.npy")
        if self.calculate_blocking_all:
            self.blocking_all = np.load(f"{self.full_data_path}/blocking_all.npy")

class ImportanceSampling(_RBMVMC):
    def __init__(
        self,
        n_particles: int,
        n_dims: int,
        n_hidden: int,
        n_mc_cycles: int,
        max_iterations: int,
        learning_rate: Union[float, str, list, dict],
        sigma: float,
        interaction: bool,
        omega: float,
        diffusion_coeff: float,
        time_step: float,
        parent_data_directory: Union[None, str] = None,
        rng_seed: Union[None, int] = None
    ) -> None:
        """
        Parameters
        ----------
        diffusion_coeff:
            The diffusion coefficient in Greens function.

        time_step:
            The importance sampling time step size.

        All other parameters are described in the super class.
        """

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
            omega,
            parent_data_directory,
            rng_seed
        )

    def reset_state_addition(self):
        """
        Reset state parameters specific to importance sampling.
        """
        self.pos_current = self.rng.normal(loc=0.0, scale=0.001, size=(self.n_particles, self.n_dims))
        self.pos_current *= np.sqrt(self.time_step)

        self.qforce_current = other.quantum_force(
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.sigma
        )

    def monte_carlo(self):
        """
        Perform the VMC using importance sampling. The actual work is
        put in a separate staticmethod to be able to compile the VMC
        loop with numba.njit. Unpacking of instance attributes is for
        numba compatibility.
        """
        n_mc_cycles = self.n_mc_cycles
        n_particles = self.n_particles
        time_step = self.time_step
        diffusion_coeff = self.diffusion_coeff
        sigma = self.sigma
        interaction = self.interaction
        omega = self.omega
        
        pre_drawn_pos_new = self.rng.normal(
            loc = 0,
            scale = 1,
            size = (self.n_particles, self.n_dims, self.n_mc_cycles)
        )*np.sqrt(self.time_step)
        
        pre_drawn_metropolis = \
            self.rng.uniform(size=(self.n_mc_cycles, self.n_particles))
        
        interaction_int = 1 if interaction else 0   # Datatype fix for constants array
        constants = np.array(
            [n_mc_cycles, n_particles, time_step, diffusion_coeff, sigma,
            interaction_int, omega]
        )

        self.monte_carlo_importance_numba(
            self.pos_new,
            self.pos_current,
            self.qforce_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.wave_current,
            self.wave_derivative_average_wrt_visible_bias,
            self.wave_derivative_average_wrt_hidden_bias,
            self.wave_derivative_average_wrt_weights,
            self.wave_derivative_energy_average_wrt_visible_bias,
            self.wave_derivative_energy_average_wrt_hidden_bias,
            self.wave_derivative_energy_average_wrt_weights,
            self.energy_mc,
            self.acceptance_rate,
            self.local_energy_average,
            pre_drawn_pos_new,
            pre_drawn_metropolis,
            constants
        )

        # All this packing and un-packing is here to make numba happy
        # TODO: Dont need the lists.
        # self.wave_derivatives_average[0] = self.wave_derivative_average_wrt_visible_bias
        # self.wave_derivatives_average[1] = self.wave_derivative_average_wrt_hidden_bias
        # self.wave_derivatives_average[2] = self.wave_derivative_average_wrt_weights
        # self.wave_derivatives_energy_average[0] = self.wave_derivative_energy_average_wrt_visible_bias
        # self.wave_derivatives_energy_average[1] = self.wave_derivative_energy_average_wrt_hidden_bias
        # self.wave_derivatives_energy_average[2] = self.wave_derivative_energy_average_wrt_weights

        self.acceptance_rate = self.acceptance_rate[0]
        self.acceptance_rate /= self.n_mc_cycles*self.n_particles
        self.local_energy_average = self.local_energy_average[0]
        self.local_energy_average /= self.n_mc_cycles
        
        self.wave_derivative_energy_average_wrt_visible_bias /= self.n_mc_cycles
        self.wave_derivative_energy_average_wrt_hidden_bias /= self.n_mc_cycles
        self.wave_derivative_energy_average_wrt_weights /= self.n_mc_cycles
        
        self.wave_derivative_average_wrt_visible_bias /= self.n_mc_cycles
        self.wave_derivative_average_wrt_hidden_bias /= self.n_mc_cycles
        self.wave_derivative_average_wrt_weights /= self.n_mc_cycles

        self.visible_biases_gradient = \
            2*(self.wave_derivative_energy_average_wrt_visible_bias - \
            self.wave_derivative_average_wrt_visible_bias*self.local_energy_average)
        self.hidden_biases_gradient = \
            2*(self.wave_derivative_energy_average_wrt_hidden_bias - \
            self.wave_derivative_average_wrt_hidden_bias*self.local_energy_average)
        self.weights_gradient = \
            2*(self.wave_derivative_energy_average_wrt_weights - \
            self.wave_derivative_average_wrt_weights*self.local_energy_average)
    
    @staticmethod
    @numba.njit
    def monte_carlo_importance_numba(
        pos_new: np.ndarray,
        pos_current: np.ndarray,
        qforce_current: np.ndarray,
        visible_biases: np.ndarray,
        hidden_biases: np.ndarray,
        weights: np.ndarray,
        wave_current: np.ndarray,
        wave_derivative_average_wrt_visible_bias: np.ndarray,
        wave_derivative_average_wrt_hidden_bias: np.ndarray,
        wave_derivative_average_wrt_weights: np.ndarray,
        wave_derivative_energy_average_wrt_visible_bias: np.ndarray,
        wave_derivative_energy_average_wrt_hidden_bias: np.ndarray,
        wave_derivative_energy_average_wrt_weights: np.ndarray,
        energy_mc: np.ndarray,
        acceptance_rate: np.ndarray,
        local_energy_average: np.ndarray,
        pre_drawn_pos_new: np.ndarray,
        pre_drawn_metropolis: np.ndarray,
        constants: np.ndarray
    ):  
        """
        Perform the Monte Carlo work for the importance sampling
        implementation. This function is broken out of the class to be numba
        compatible, and is therefore a bit ugly in regards of input
        arguments and breaking of the class structure (file a complaint to
        the numba developers, not me!).

        All calculations in this function operate on arrays which is why
        there are no return values. See RBMVMC class documentation for
        description of all these input parameters.
        """
        n_mc_cycles, n_particles, time_step, diffusion_coeff, sigma, interaction, omega = constants
        for cycle in range(int(n_mc_cycles)):
            for particle in range(int(n_particles)):
                """
                Loop over all particles. Move one particle at the time.
                """
                pos_new[particle] = pos_current[particle]
                pos_new[particle] += pre_drawn_pos_new[particle, :, cycle]
                pos_new[particle] += qforce_current[particle]*time_step*diffusion_coeff

                wave_new = other.wave_function(
                    pos_new,
                    visible_biases,
                    hidden_biases,
                    weights,
                    sigma
                )

                qforce_new = other.quantum_force(
                    pos_new,
                    visible_biases,
                    hidden_biases,
                    weights,
                    sigma
                )

                greens_function = 0.5*(qforce_current[particle] + qforce_new[particle])
                greens_function *= (diffusion_coeff*time_step*0.5*(qforce_current[particle] - qforce_new[particle]) - pos_new[particle] + pos_current[particle])
                greens_function = np.exp(greens_function.sum())

                if pre_drawn_metropolis[cycle, particle] <= greens_function*(wave_new/wave_current)**2:
                    """
                    Metropolis-Hastings.
                    """
                    acceptance_rate[0] += 1
                    pos_current[particle] = pos_new[particle]
                    qforce_current[particle] = qforce_new[particle]
                    wave_current = wave_new

            local_energy_partial = other.local_energy(
                pos_current,
                visible_biases,
                hidden_biases,
                weights,
                sigma,
                interaction,
                omega
            )

            wrt_visible_bias_tmp, wrt_hidden_bias_tmp, wrt_weights_tmp = other.wave_function_derivative(
                pos_current,
                visible_biases,
                hidden_biases,
                weights,
                sigma
            )

            wave_derivative_average_wrt_visible_bias += wrt_visible_bias_tmp
            wave_derivative_average_wrt_hidden_bias += wrt_hidden_bias_tmp
            wave_derivative_average_wrt_weights += wrt_weights_tmp

            local_energy_average[0] += local_energy_partial
            energy_mc[cycle] = local_energy_partial

            wave_derivative_energy_average_wrt_visible_bias += local_energy_partial*wrt_visible_bias_tmp
            wave_derivative_energy_average_wrt_hidden_bias += local_energy_partial*wrt_hidden_bias_tmp
            wave_derivative_energy_average_wrt_weights += local_energy_partial*wrt_weights_tmp

    def __str__(self):
        return "Importance sampling"

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
        brute_force_step_size: float,
        parent_data_directory: Union[None, str] = None,
        rng_seed: Union[int, None] = None
    ) -> None:
        """
        Parameters
        ----------
        brute_force_step_size:
            The brute-force step size.

        All other parameters are described in the super class.
        """
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
            omega,
            parent_data_directory,
            rng_seed
        )

    def reset_state_addition(self) -> None:
        self.pos_current = self.rng.uniform(low=-0.5, high=0.5, size=(self.n_particles, self.n_dims))*self.brute_force_step_size

    @staticmethod
    @numba.njit
    def monte_carlo_brute_numba(
        pos_new: np.ndarray,
        pos_current: np.ndarray,
        visible_biases: np.ndarray,
        hidden_biases: np.ndarray,
        weights: np.ndarray,
        wave_current: np.ndarray,
        wave_derivative_average_wrt_visible_bias: np.ndarray,
        wave_derivative_average_wrt_hidden_bias: np.ndarray,
        wave_derivative_average_wrt_weights: np.ndarray,
        wave_derivative_energy_average_wrt_visible_bias: np.ndarray,
        wave_derivative_energy_average_wrt_hidden_bias: np.ndarray,
        wave_derivative_energy_average_wrt_weights: np.ndarray,
        energy_mc: np.ndarray,
        acceptance_rate: np.ndarray,
        local_energy_average: np.ndarray,
        pre_drawn_pos_new: np.ndarray,
        pre_drawn_metropolis: np.ndarray,
        constants: np.ndarray
    ):
        """
        Perform the Monte Carlo work for the brute force implementation.
        This function is broken out of the class to be numba compatible,
        and is therefore a bit ugly in regards of input arguments and
        breaking of the class structure (file a complaint to the numba
        developers, not me!).

        All calculations in this function operate on arrays which is why
        there are no return values. See RBMVMC class documentation for
        description of all these input parameters.
        """
        n_mc_cycles, n_particles, sigma, interaction, omega = constants
        for cycle in range(int(n_mc_cycles)):
            for particle in range(int(n_particles)):
                """
                Loop over all particles. Move one particle at the time.
                """
                pos_new[particle] = pos_current[particle]
                pos_new[particle] += pre_drawn_pos_new[particle, :, cycle]

                wave_new = other.wave_function(
                    pos_new,
                    visible_biases,
                    hidden_biases,
                    weights,
                    sigma
                )

                if pre_drawn_metropolis[cycle, particle] <= (wave_new/wave_current)**2:
                    """
                    Metropolis-Hastings.
                    """
                    acceptance_rate[0] += 1
                    pos_current[particle] = pos_new[particle]
                    wave_current = wave_new

            local_energy_partial = other.local_energy(
                pos_current,
                visible_biases,
                hidden_biases,
                weights,
                sigma,
                interaction,
                omega
            )

            wrt_visible_bias_tmp, wrt_hidden_bias_tmp, wrt_weights_tmp = \
                other.wave_function_derivative(
                    pos_current,
                    visible_biases,
                    hidden_biases,
                    weights,
                    sigma
                )

            wave_derivative_average_wrt_visible_bias += wrt_visible_bias_tmp
            wave_derivative_average_wrt_hidden_bias += wrt_hidden_bias_tmp
            wave_derivative_average_wrt_weights += wrt_weights_tmp

            local_energy_average[0] += local_energy_partial
            energy_mc[cycle] = local_energy_partial

            wave_derivative_energy_average_wrt_visible_bias += \
                local_energy_partial*wrt_visible_bias_tmp
            wave_derivative_energy_average_wrt_hidden_bias += \
                local_energy_partial*wrt_hidden_bias_tmp
            wave_derivative_energy_average_wrt_weights += \
                local_energy_partial*wrt_weights_tmp

    def monte_carlo(self) -> None:
        # For numba compatibility:
        n_mc_cycles = self.n_mc_cycles
        n_particles = self.n_particles
        sigma = self.sigma
        interaction = self.interaction
        omega = self.omega
        
        pre_drawn_pos_new = self.rng.uniform(
            low = -0.5,
            high = 0.5,
            size = (self.n_particles, self.n_dims, self.n_mc_cycles)
        )*self.brute_force_step_size
        
        pre_drawn_metropolis = \
            self.rng.uniform(size=(self.n_mc_cycles, self.n_particles))
        
        interaction_int = 1 if interaction else 0   # Datatype fix for constants array
        constants = np.array(
            [n_mc_cycles, n_particles, sigma, interaction_int, omega]
        )

        self.monte_carlo_brute_numba(
            self.pos_new,
            self.pos_current,
            self.visible_biases,
            self.hidden_biases,
            self.weights,
            self.wave_current,
            self.wave_derivative_average_wrt_visible_bias,
            self.wave_derivative_average_wrt_hidden_bias,
            self.wave_derivative_average_wrt_weights,
            self.wave_derivative_energy_average_wrt_visible_bias,
            self.wave_derivative_energy_average_wrt_hidden_bias,
            self.wave_derivative_energy_average_wrt_weights,
            self.energy_mc,
            self.acceptance_rate,
            self.local_energy_average,
            pre_drawn_pos_new,
            pre_drawn_metropolis,
            constants
        )

        # All this packing and un-packing is here to make numba happy
        self.wave_derivatives_average[0] = self.wave_derivative_average_wrt_visible_bias
        self.wave_derivatives_average[1] = self.wave_derivative_average_wrt_hidden_bias
        self.wave_derivatives_average[2] = self.wave_derivative_average_wrt_weights
        self.wave_derivatives_energy_average[0] = self.wave_derivative_energy_average_wrt_visible_bias
        self.wave_derivatives_energy_average[1] = self.wave_derivative_energy_average_wrt_hidden_bias
        self.wave_derivatives_energy_average[2] = self.wave_derivative_energy_average_wrt_weights

        self.acceptance_rate = self.acceptance_rate[0]
        self.acceptance_rate /= self.n_mc_cycles*self.n_particles
        self.local_energy_average = self.local_energy_average[0]
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

    def __str__(self):
        return "Brute force"

def main():
    """
    The content of this function is for testing purposes. All actual
    runs are administered via separate files.
    """
    # self.brute_force_step_size = 0.05
    # omega = 1/4
    omega = 1
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 1,
        n_dims = 1,
        n_hidden = 2,
        n_mc_cycles = int(2**12),
        max_iterations = 50,
        learning_rate = 0.1,
        # learning_rate = {"factor": 0.1, "init": 0.05},
        sigma = sigma,
        interaction = False,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        # rng_seed = 1337
    )
    # q = BruteForce(
    #     n_particles = 2,
    #     n_dims = 2,
    #     n_hidden = 2,
    #     n_mc_cycles = int(2**16),
    #     max_iterations = 100,
    #     learning_rate = 0.5,
    #     # learning_rate = {"factor": 0.1, "init": 0.05},
    #     sigma = sigma,
    #     interaction = True,
    #     omega = omega,
    #     brute_force_step_size = 1,
    #     rng_seed = 1337
    # )
    q.initial_state(
        loc_scale_hidden_biases = (0, 0.5),
        loc_scale_visible_biases = (0, 0.5),
        loc_scale_weights = (0, 0.5)
    )
    q.solve(verbose=True, save_state=False)
    # print(q.energies[-1])
    # import matplotlib.pyplot as plt
    # plt.plot(range(q.max_iterations), q.energies)
    # plt.show()

if __name__ == "__main__":
    main()
