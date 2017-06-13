#include "book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define N   64*1024*1024

__global__ void add( int *a, int *b, int *c, int seg_size ) {
    int tid = (threadIdx.x + blockIdx.x * blockDim.x) * seg_size;
    while (tid < N) {
	for(int i=tid; i<tid+seg_size; i++){
        	c[i] = a[i] + b[i];
	}
        
        tid += (blockDim.x * gridDim.x)*seg_size;
    }
}

int main( int argc, char* argv[] ) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    if(argc!=2){
	printf("using command %s [segment_size]\n", argv[0]);
 	exit(1);
    }
    // allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
    c = (int*)malloc( N * sizeof(int) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = 2 * i;
	c[i] = 0;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    // Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int threadsPerBlock = 256;
    int segment_size = atoi(argv[1]);
    int segment_num = (N + segment_size -1)/segment_size;
    int blocksPerGrid = (segment_num + threadsPerBlock -1)/threadsPerBlock;
    if(blocksPerGrid > 65535)
	blocksPerGrid = 65535;
    printf("threadsPerBlock: %d\n", threadsPerBlock);
    printf("segment_size: %d\n", segment_size);
    printf("segment_number: %d\n", segment_num);
    printf("blocksPerGrid: %d\n", blocksPerGrid);


    add<<<blocksPerGrid, threadsPerBlock>>>( dev_a, dev_b, dev_c, segment_size );
	 
	// Get stop time event    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
	//check cuda error
    cudaError_t status = cudaGetLastError();
    if ( cudaSuccess != status ){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
        exit(1) ;
    }
	
    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // verify that the GPU did the work we requested
    bool success = true;
    for (int i=0; i<N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf( "Error:  %d + %d != %d\n", a[i], b[i], c[i] );
            success = false;
        }
    }
    if (success)    printf( "We did it!\n" );

    // free the memory we allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );

    // free the memory we allocated on the CPU
    free( a );
    free( b );
    free( c );

    return 0;
}

