CC=nvcc 

RTIOW.out:  mkClockMeasure.h  mkCuda.h main.cu 
	$ rm -rf main.out
	$(CC) -o main.out main.cu
	$ ./main.out > main.txt


clean:
	rm -rf RTIOW.out
