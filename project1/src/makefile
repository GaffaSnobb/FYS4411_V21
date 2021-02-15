all : main.out

main.out :
	g++ -std=c++17 -o run.out main.cpp -O1 -larmadillo -fopenmp

valgrind : main.out
	valgrind --tool=memcheck ./run.out

make plot : main.out
	./run.out
	python visualize.py

clean :
	rm *.out
