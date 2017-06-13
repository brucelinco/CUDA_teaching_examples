/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 *1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
            imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );


__global__ void dot( int *a, int *b, int *c ) {
    __shared__ int cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
}
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main( void ) {
    int   *a, *b, c, *partial_c;
    int   *dev_a, *dev_b, *dev_partial_c;
	struct timespec t_start, t_end;
	int i;
    // allocate memory on the cpu side
    a = (int*)malloc( N*sizeof(int) );
    b = (int*)malloc( N*sizeof(int) );
    partial_c = (int*)malloc( blocksPerGrid*sizeof(int) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
                              N*sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b,
                              N*sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c,
                              blocksPerGrid*sizeof(int) ) );

    // fill in the host memory with data
    for (i=0; i<N; i++) {
        a[i] = rand()%256;
        b[i] = rand()%256;
    }
    // Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(int),
                              cudaMemcpyHostToDevice ) ); 

	
    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,
                                            dev_partial_c );
	

	
	//check cuda error
    cudaError_t status = cudaGetLastError();
    if ( cudaSuccess != status ){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
        exit(1) ;
    }											

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( partial_c, dev_partial_c,
                              blocksPerGrid*sizeof(int),
                              cudaMemcpyDeviceToHost ) );
	// Get stop time event    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
    // finish up on the CPU side
    c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }
    
	//printf("GPU result is %d\n",c);
    // start time
    clock_gettime( CLOCK_REALTIME, &t_start);
	/*CPU version*/
	int dot=0;
	for(i=0;i<N;i++){
	   dot+=a[i]*b[i];
	}
	// stop time
    clock_gettime( CLOCK_REALTIME, &t_end);

    // compute and print the elapsed time in millisec
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("CPU time: %13lf ms\n", elapsedTime);
	//printf("CPU result is %d\n",dot);
	
	if(c==dot)
	   printf("test pass!\n");
	else
	   printf("test fail!\n");
	
    // free memory on the gpu side
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_partial_c ) );

    // free memory on the cpu side
    free( a );
    free( b );
    free( partial_c );
}
