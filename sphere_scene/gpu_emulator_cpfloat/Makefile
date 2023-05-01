CC=nvcc 


main.out: mkClockMeasure.h mkCuda.h main.cu
	$(CC) --expt-relaxed-constexpr -rdc=true -lcudadevrt -o main_5_10.out main.cu

clean:
	rm -rf ./main_5_10.out