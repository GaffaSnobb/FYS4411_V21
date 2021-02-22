all : main.out

main.out : VMC.o wave_function.o other_functions.o
	g++ -std=c++17 -o run.out main.cpp VMC.o wave_function.o other_functions.o -O1 -larmadillo -fopenmp
# main.out : wave_function.o other_functions.o
# 	g++ -std=c++17 -o run.out main.cpp wave_function.o other_functions.o -O1 -larmadillo -fopenmp

VMC.o : VMC.h VMC.cpp
	g++ -c VMC.cpp -std=c++17 -O1

wave_function.o : wave_function.h wave_function.cpp
	g++ -c wave_function.cpp -std=c++17 -O1

other_functions.o : other_functions.h other_functions.cpp
	g++ -c other_functions.cpp -std=c++17 -O1

valgrind : main.out
	valgrind --tool=memcheck ./run.out

make plot : main.out
	./run.out
	python visualize.py

.PHONY : clean
clean :
	-rm *.out
	-rm *.o
	-rm -r __pycache__/
