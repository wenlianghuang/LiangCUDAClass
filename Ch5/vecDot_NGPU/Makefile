################################################################################
#
# Build script for project
#
################################################################################

CXX     := icc
CFLAGS  := -O3
NVCC    := nvcc
#CUFLAGS := -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true

CUFLAGS := -arch=compute_52 -code=sm_52,sm_52 -O3 --compiler-options -fopenmp
#CUFLAGS := -O3 --compiler-options -fopenmp

BIN     := vecDot_NGPU
OBJ     := vecDot_NGPU.o
LIB     :=


all: $(BIN)

$(BIN): $(OBJ)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

