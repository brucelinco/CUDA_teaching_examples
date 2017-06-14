#include "book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define N   1024*1024

__global__ void add( int *colors, int *bucket ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        int c = colors[tid];
		atomicAdd(&bucket[c], 1);
		//bucket[c]++;
        tid += blockDim.x * gridDim.x;	
    }
}

int main( void ) {
    int *colors;
	int bucket[256]={0};
	int golden[256]={0};
    int *dev_colors, *dev_bucket;
    int i;
 
 // allocate the memory on the CPU
    colors = (int*)malloc( N * sizeof(int) );
 
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_colors, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bucket, 256 * sizeof(int) ) );


    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        colors[i] = rand()%256;
		golden[colors[i]]++;
    }
    
	
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_colors, colors, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_bucket, bucket, 256 * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    // Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
    add<<<65535,256>>>( dev_colors, dev_bucket );
	 
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
    HANDLE_ERROR( cudaMemcpy( bucket, dev_bucket, 256 * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // verify that the GPU did the work we requested
    bool success = true;
    for (int i=0; i<256; i++) {
        if (golden[i] != bucket[i]) {
            printf( "Error at bucket[%d]\n", i);
            success = false;
        }
    }
    if (success)    printf( "We did it!\n" );

    // free the memory we allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_colors ) );
    HANDLE_ERROR( cudaFree( dev_bucket ) );
 

    // free the memory we allocated on the CPU
    free( colors );


    return 0;
}

