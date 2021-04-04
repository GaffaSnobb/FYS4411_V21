# A Variational Monte Carlo Analysis of Bose-Einstein Condensation

The ground state energy of hard-shpere bosons is calculated using Variational Monte Carlo methods. Two applications of the Metropolis sampling method is used, first the so-called brute force metropolis method before adding importance sampling based on the Langevin and Fokker-Planck equations. We will also appply the blocking method for statistical analysis of our results.

Authors: Jon Kristian Dahl, Alida Hardersen, and Per-Dimitri Sønderland

## Folders

- src: contains c++ code and header files, as well as the makefile for compiling and running.
- scripts: contains python scripts for plotting and a statistical analysis using the blocking method.
- fig: contains figures generated.

## Dependencies

Install armadillo: http://arma.sourceforge.net/

Python version 3.8


## How to use
### 1. Compile and run
The program uses make to compile and run. Make sure that the 'COMPILER' in 'src/makefile' is correct. Then, move to the src folder and compile by writing in a terminal window,

```
$ make
```

Or you can compile and run by writing

```
$ make run
```

To change the method, edit lines 132-133 in 'main.cpp':

```
// Select methods (choose one at a time):
const bool gradient_descent    = false;
const bool importance_sampling = true;
const bool brute_force         = false;

```

To vary other parameters such as step size (brute force) and time step (importance), number of variations, number of mc-cycles etc. edit the global parameters in `main.cpp` lines 113 - 130.

To recompile the programs it may be useful to first remove the old compilation files, this can easily be done by the command,

```
$ make clean
```

## Credit
The block() function in `src/blocking.py` is based on the blocking code written by Marius Jonsson for the paper:

- Jonsson, M. (2018). Standard error estimation by an automated blocking method. Physical Review E, 98(4), 043304.


The autodiff c++ library is used for numerical differentiation.

MIT License\
Copyright © 2018–2021 Allan Leal
