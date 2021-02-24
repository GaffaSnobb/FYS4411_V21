
## Dependencies

Install armadillo: https://www.uio.no/studier/emner/matnat/fys/FYS4411/v13/guides/installing-armadillo/

## How to use
### 1. Compile the main program
The program is compiled using make. In a terminal write,
```
$ make
```

To compile, run binary, and plot the results use the command,
```
$ make plot
```
You can also run memcheck with valgrind.
```
$ make valgrind
```
Be sure to shorten the program run time by decreasing the number of MC cycles or other parameters. This is done by modifying the `n_mc_cycles` variable on line xx of `VCM.h`. Else, valgrind will use a very long time.

## Credit
The block() function in `src/blocking.py` is written by Marius Jonsson for the paper:

- Jonsson, M. (2018). Standard error estimation by an automated blocking method. Physical Review E, 98(4), 043304.

## Edits(tmp)
change time step on line 348 in main.cpp for now.
