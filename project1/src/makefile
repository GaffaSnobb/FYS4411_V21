COMPILER = g++
#COMPILER = g++-10
FLAGS = -std=c++17 -O1
LIBRARIES = -larmadillo -fopenmp
OBJECTS = VMC.o wave_function.o local_energy.o methods.o quantum_force.o parameters.o

all : main.out

test : arma_test.cpp
	g++-10 arma_test.cpp -o arma_test.out -std=c++17 -larmadillo

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

parameters.o : parameters.h parameters.cpp
	$(COMPILER) $(FLAGS) -c parameters.cpp

plot : main.out
	./run.out
	python visualize.py

ploit : main.out	# Alias.
	./run.out
	python visualize.py

run : main.out
	./run.out

blocking : main.out
	./run.out
	python blocking.py

.PHONY : clean
clean :
	-rm *.out
	-rm *.o
	-rm -r __pycache__/
	-rm -r run.out.dSYM/
	# -rm generated_data/*.txt
