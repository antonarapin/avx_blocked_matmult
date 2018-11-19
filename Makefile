# On Bridges we will check versus your performance versus Intel MKL library's BLAS. 

CC = gcc-6 
OPT = -O3
CFLAGS = -Wall -std=gnu99 -O3 -mavx -ftree-vectorize -fopt-info -funroll-all-loops $(OPT)
#MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
#LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
#LDLIBS = /usr/lib/libblas -lrt  -I$(MKLROOT)/include -Wl,-L$(MKLROOT)/lib/intel64/ -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
#LDLIBS = /usr/lib/libblas/libblas.so.3.0 -lpthread -lm
LDLIBS = /home/aarapin/lib/OpenBLAS-0.2.20/libopenblas.a -lpthread -lm -ldl
targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
