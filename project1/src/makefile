COMPILER = g++
# COMPILER = g++-10
FLAGS = -std=c++17 -O1
LIBRARIES = -larmadillo -fopenmp
OBJECTS = VMC.o wave_function.o local_energy.o methods.o quantum_force.o

all : main.out

main.out : $(OBJECTS)
	$(COMPILER) $(FLAGS) $(OBJECTS) $(LIBRARIES) -o run.out main.cpp

VMC.o : VMC.h VMC.cpp
	$(COMPILER) $(FLAGS) $(LIBRARIES) -c VMC.cpp

methods.o : methods.cpp methods.h
	$(COMPILER) $(FLAGS) $(LIBRARIES) -c methods.cpp

wave_function.o : wave_function.h wave_function.cpp
	$(COMPILER) $(FLAGS) -c wave_function.cpp

local_energy.o : local_energy.h local_energy.cpp
	$(COMPILER) $(FLAGS) -c local_energy.cpp

quantum_force.o : quantum_force.h quantum_force.cpp
	$(COMPILER) $(FLAGS) -c quantum_force.cpp

plot : main.out
	./run.out
	python visualize.py

blocking : main.out
	./run.out
	python blocking.py

.PHONY : clean
clean :
	-rm *.out
	-rm *.o
	-rm -r __pycache__/
	-rm generated_data/*.txt
