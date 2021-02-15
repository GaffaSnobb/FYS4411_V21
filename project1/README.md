
## Dependencies

Install armadillo: https://www.uio.no/studier/emner/matnat/fys/FYS4411/v13/guides/installing-armadillo/

## How to use
### 1. Compile the main program
The program is compiled using the `makefile`. To compile from command line use,
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
Be sure to shorten the program run time by decreasing the number of MC cycles or other parameters. This is done by modifying the `n_mc_cycles` variable on line 24 of `main.cpp`. Else, valgrind will use a very long time.

## Credit
If we borrow code etc.

## Overview of files
| Files | Description |
| ------ | ------ |
| makefile | description |
| main.cpp | description |
| other_functions.cpp | description |
| wave_function.cpp | description |
| read_from_file.py | description |

## Edits(tmp)
change time step on line 348 in main.cpp for now. 
