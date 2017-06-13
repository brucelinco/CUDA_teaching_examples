NVCC30   = /usr/local/cuda/bin/nvcc
SRC      = vecAdd_chunk.cu
INC30    = -I/usr/local/cuda/include
LIB30    = -L/usr/local/cuda/lib:../lib
ARGS13   = -arch=sm_13 -Xcudafe=-w
ARGS20   = -arch=sm_20 -Xcudafe=-w
ARGS30   = -arch=sm_30 -Xcudafe=-w
o        = vecAdd_chunk

all :: cuda20

cuda13 :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS13) -o $(o) $(SRC)
	rm -f *.linkinfo
cuda20 :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS20) -o $(o) $(SRC)
	rm -f *.linkinfo
cuda30 :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS30) -o $(o) $(SRC)
	rm -f *.linkinfo
memuse :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS13) -Xptxas=-v -o $(o) $(SRC)
emu :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS13) -deviceemu -o $(o)_emu $(SRC)
cubin :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS13) -cubin $(SRC)
clean :
	rm -f *.linkinfo
