CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v

BIN     := matMultiplication
LIB     :=

all: $(BIN)

$(BIN): matMultiplication.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) matMultiplication.cu $(LIB)

clean:
	rm -f $(BIN)
