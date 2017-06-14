#include "book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define N   1024*1024

__global__ void global_max( int *d_values, int *d_global_max ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = d_values[tid];
	atomicMax(d_global_max, val);
}

int main( void ) {
    int *values;
	int globalMax;
	int golden_globalMax = 0;
    int *dev_values, *dev_globalMax;
    int i;
 
 // allocate the memory on the CPU
    values = (int*)malloc( N * sizeof(int) );
 
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_values, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_globalMax, sizeof(int) ) );


    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        values[i] = rand();
		if(values[i] > golden_globalMax)
			golden_globalMax = values[i];
    }
    
	
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_values, values, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    // Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
    global_max<<<N/256,256>>>( dev_values, dev_globalMax );
	 
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
    HANDLE_ERROR( cudaMemcpy( &globalMax, dev_globalMax, sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // verify that the GPU did the work we requested
    bool success = true;
    if (globalMax != golden_globalMax) {
        printf( "globalMax:%d, golden_globalMax: %d\n", globalMax, golden_globalMax);
        success = false;
    }
 
    if (success)    printf( "We did it!\n" );

    // free the memory we allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_values ) );
    HANDLE_ERROR( cudaFree( dev_globalMax ) );
 

    // free the memory we allocated on the CPU
    free( values );


    return 0;
}

