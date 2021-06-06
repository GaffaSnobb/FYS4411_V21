# FYS4411 - The Restricted Boltzmann Machine Applied to the Quantum Many-Body Problem

In this project we will apply a restricted Boltzmann machine (RBM) to a system of interacting electrons confined in a harmonic oscillator trap. The parameters of the RBM are optimized using variational Monte Carlo (VMC), where we employ the Metropolis-Hastings algorithm to sample particle positions. We will be using both the brute-force approach as well as the addition of importance sampling, and present a comparison of the performance of the two methods. The minimization method we will be using in the VMC is a gradient descent (GD) method based on the variational principle which says that minimizing the energy of the quantum mechanical system should lead to the ground state wave function.

## Dependencies

- Python [3.8.5]
- Numpy [1.19.4]
- Matplotlib [3.3.3]
- Numba [0.53.1]
- Pytest [6.2.3]

## Usage

The RBM is located in the 'boltzmann_machine.py' script. For improved readability, several functions have been put in a separate file, ```other_functions.py```. The superclass ```_RBMVMC``` contains common tools for both importance sampling and brute-force, like ```_load_state``` and ```_save_state``` methods and a common gradient descent method, ```solve```. The subclasses ```ImportanceSampling``` and ```BruteForce``` inherit from ```_RBMVMC``` and they contain the VMC loop where the Metopolis-Hastings algorithm is located. All additional Python files contain specific parameters for the RBM. Any of the files may be run to produce results which we include in the report. Run by ```python file.py```. All of the functions which contain heavy work are compiled with the ```@numba.njit``` decorator.  Here is an example of use:

``` python
q = BruteForce(
      n_particles = 1,
      n_dims = 1,
      n_hidden = 4,
      n_mc_cycles = int(2**10),
      max_iterations = 100,
      learning_rate = 0.1,
      sigma = 1,
      interaction = False,
      omega = 1,
      brute_force_step_size = 1,
      parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
      rng_seed = 1337
  )

  q.initial_state(
      loc_scale_all = (0, 1)
  )

  q.solve(verbose=False)

```
Most notable attributes:
``` python
q.energy_mc_iter    # Energies for all MC cycles and all GD iterations
q.acceptance_rates  # Acceptance rate per GD iteration
q.energies          # Energy per GD iteration
q.times             # Time per GD iteration
q.blocking_final    # Blocking uncertainty for all GD iterations
q.blocking_all      # Blocking uncertainty for last GD iteration
```
These attributes are all numpy arrays and are saved after calculation. If the RBM is run with the same configuration, the files are loaded instead of re-calculated. To force re-calculation, delete the correct directory in tmp/ or the entire tmp/ directory.

Each initial distribution can be altered separately:
``` python
q.initial_state(
    loc_scale_visible_biases = (0, 0.1),
    loc_scale_hidden_biases = (0, 0.1),
    loc_scale_weights = (0, 0.1)
)
```

It is possible to skip saving and / or loading files:
``` python
q.solve(
    save_state = False,
    load_state = False
)
```
If ```load_state``` is set to ```False```, then ```save_state``` is forced to ```False```.

```test_multiple.py``` contains three tests:
- Check that two different runs with the same input RNG seed draws the same random numbers.
- Check that two different runs with different RNG seeds draws different random numbers.
- Check that both importance sampling and brute-force manages to reproduce analytical answers for one and two particles in one, two, and three dimensions, within a tolerance.

Run the tests with ```pytest```.

## Credit

Code is based on example code found at: [http://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/boltzmannmachines.html#representing-the-wave-function]
