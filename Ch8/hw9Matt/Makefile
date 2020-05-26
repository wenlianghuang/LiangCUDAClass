CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O2 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v -rdc=true --compiler-options -fopenmp
#NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true --std=c++11
LIB     := -lgsl -lgslcblas -lcudart 


BIN1   := ising2d_1gpu_gmem_v2
BIN2   := ising2d_Ngpu_gmem


all: $(BIN1) $(BIN2)

$(BIN1): ising2d_1gpu_gmem_v2.cu
	$(NVCC) -o $(BIN1) $(NVFLAGS) $< $(LIB)

$(BIN2): ising2d_Ngpu_gmem.cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) $< $(LIB)



clean:
	rm -f $(BIN1) $(BIN2) *.o

