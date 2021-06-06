# FYS4411 - [project title]

In this project we will apply a restricted Boltzmann machine (RBM) to a system of interacting electrons confined in a harmonic oscillator trap. The parameters of the RBM are optimized using variational Monte Carlo (VMC), where we employ the Metropolis-Hastings algorithm to sample particle positions. We will be using both the brute-force approach as well as the addition of importance sampling, and present a comparison of the performance of the two methods. The minimization method we will be using in the VMC is a gradient descent (GD) method based on the variational principle which says that minimizing the energy of the quantum mechanical system should lead to the ground state wave function.

## Dependencies

- Python [3.8.5]
- Numpy [1.19.4]
- Matplotlib [3.3.3]
- Numba [0.53.1]
- Pytest [6.2.3]

## Usage

The RBM is located in the 'boltzmann_machine.py' script. Here is an example of use:

```
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
```
q.energy_mc_iter
q.acceptance_rates
q.energies
q.times
q.blocking_final
q.blocking_all
```

## Credit

Code is based on example code found at: [http://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/boltzmannmachines.html#representing-the-wave-function]
