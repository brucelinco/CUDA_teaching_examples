NVCC   = nvcc
SRC    = *.c*
INC    = -I/usr/local/cuda/include
LIB    = -L/usr/local/cuda/lib64
ARGS   = -arch=sm_13 -Xcudafe=-w
o      = cuda

all :: compile

compile :
	$(NVCC) $(INC) $(LIB) $(ARGS) -o $(o) $(SRC)
memuse :
	$(NVCC) $(INC) $(LIB) $(ARGS) -Xptxas=-v -o $(o) $(SRC)
emu :
	$(NVCC) $(INC) $(LIB) $(ARGS) -deviceemu -o $(o)_emu $(SRC)
cubin :
	$(NVCC) $(INC) $(LIB) $(ARGS) -cubin $(SRC)
