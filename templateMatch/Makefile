NVCC30   = /usr/local/cuda/bin/nvcc
INC30	 = -I/usr/local/cuda/include
LIB30    = -L/usr/local/cuda/lib64
ARGS20   = -arch=sm_30 -Xcudafe=-w -lrt
INC_OpenCV	= -I/usr/include/opencv
ARGS_openCV = -lopencv_core -lopencv_imgproc -lopencv_highgui
obj_GPU    = templateMatch_gpu
obj_CPU	 = templateMatch
SRC_CPU  = templateMatch.c
SRC_GPU	 = templateMatch_gpu.cu

all :: cpu
cpu :
	$(NVCC30) $(INC_OpenCV) $(ARGS_openCV) $(SRC_CPU) -o $(obj_CPU)
gpu :
	$(NVCC30) $(INC30) $(ARGS20) $(SRC_GPU) -o $(obj_GPU) $(ARGS_openCV)
	rm -f *.linkinfo
ld: 	
	$(NVCC30) $(LIB30) $(obj_CPU) $(obj_GPU) $(ARGS_openCV) -o tm 
memuse :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS20) -Xptxas=-v -o $(o) $(SRC)
cubin :
	$(NVCC30) $(INC30) $(LIB30) $(ARGS20) -cubin $(SRC)
clean :
	rm -f *.linkinfo
