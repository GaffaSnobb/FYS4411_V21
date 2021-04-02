
## Dependencies

Install armadillo: https://www.uio.no/studier/emner/matnat/fys/FYS4411/v13/guides/installing-armadillo/

Python version 3.8

## How to use
### 1. Compile the main program
The program uses the makefile to compile and run. Make sure that the 'COMPILER' in the makefile is correct. Then, in a terminal write,

```
$ make
```

To change the sampling method, edit the lines 132-133 in 'main.cpp':

```
// Select methods (choose one at a time):
const bool gradient_descent    = false;
const bool importance_sampling = true;
const bool brute_force         = false;

```
To vary the parameters such as step size (brute force) and time step (importance), number of variations, number of mc-cycles etc. edit the global parameters in `main.cpp` lines 113 - 130.


## Credit
The block() function in `src/blocking.py` is strongly based on the blocking code written by Marius Jonsson for the paper:

- Jonsson, M. (2018). Standard error estimation by an automated blocking method. Physical Review E, 98(4), 043304.


The autodiff c++ library is used for numerical differentiation.\

MIT License\
Copyright © 2018–2021 Allan Leal
