COMPILER = g++
# COMPILER = g++-10
FLAGS = -std=c++17 -O1
LIBRARIES = -larmadillo# -fopenmp
OBJECTS = VMC.o wave_function.o other_functions.o methods.o

all : main.out

main.out : $(OBJECTS)
	$(COMPILER) $(FLAGS) $(OBJECTS) $(LIBRARIES) -o run.out main.cpp

VMC.o : VMC.h VMC.cpp
	$(COMPILER) $(FLAGS) -c VMC.cpp

methods.o : methods.cpp methods.h
	$(COMPILER) $(FLAGS) -c methods.cpp

wave_function.o : wave_function.h wave_function.cpp
	$(COMPILER) $(FLAGS) -c wave_function.cpp

other_functions.o : other_functions.h other_functions.cpp
	$(COMPILER) $(FLAGS) -c other_functions.cpp

valgrind : main.out
	valgrind --tool=memcheck ./run.out

make plot : main.out
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
