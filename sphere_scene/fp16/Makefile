CC=nvcc 

RTTNW.out:  mkClockMeasure.h  mkCuda.h mkPpm.h main.cu 
	$(CC) -o gpu.out -arch=sm_86  main.cu
	$ ./gpu.out

clean:
	rm -rf gpu.out
	rm *.ppm
