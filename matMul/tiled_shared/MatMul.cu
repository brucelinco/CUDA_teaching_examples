#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define WIDTH       512  // 64 ~ 512
#define TILE_WIDTH  16

#define ARYTYPE float

ARYTYPE M[WIDTH][WIDTH] = {0};
ARYTYPE N[WIDTH][WIDTH] = {0};
ARYTYPE P[WIDTH][WIDTH] = {0};
ARYTYPE MxN[WIDTH][WIDTH] = {0};

__device__ ARYTYPE GetElement(ARYTYPE *matrix, int row, int col, int width);
__device__ void SetElement(ARYTYPE *matrix, int row, int col, int width, ARYTYPE value);
__device__ ARYTYPE *GetSubMatrix(ARYTYPE *matrix, int blockrow, int blockcol, int width);
__global__ void MatMulKernel(ARYTYPE *Md, ARYTYPE *Nd, ARYTYPE *Pd, int width);
void MatMul(ARYTYPE *M, ARYTYPE *N, ARYTYPE *P, int width);


int main(int argc, char *argv[])
{
    int width = WIDTH;
    int pass = 1;
    
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            M[i][j] = rand() % 30;
            N[i][j] = rand() % 30;
        }
    }
    
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < width; ++k) {
                MxN[i][j] += M[i][k] * N[k][j];
            }
        }
    }
    gettimeofday(&endtime, NULL);
    double executime;
    executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
    executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
    printf("CPU time: %13lf msec\n", executime);
    
    MatMul((ARYTYPE *)M, (ARYTYPE *)N, (ARYTYPE *)P, width);
    
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if(MxN[i][j] != P[i][j]) {
				printf("MxN[%d][%d] = %2.0f   P[%d][%d] = %2.0f\n", i, j, (float)MxN[i][j], i, j, (float)P[i][j]);
                pass = 0;
            }
        }
    }
    
    printf("Test %s\n", (pass)?"PASSED":"FAILED");
    
    return 0;
}

// Get a matrix element
__device__ ARYTYPE GetElement(ARYTYPE *matrix, int row, int col, int width)
{
    return *(matrix + row*width + col);
}

// Set a matrix element
__device__ void SetElement(ARYTYPE *matrix, int row, int col, int width, ARYTYPE value)
{
    *(matrix + row*width + col) = value;
}

// Get the TILE_WIDTHxTILE_WIDTH sub-matrix matsub of matrix that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of matrix
__device__ ARYTYPE *GetSubMatrix(ARYTYPE *matrix, int blockrow, int blockcol, int width)
{
    return (matrix + blockrow*TILE_WIDTH*width + blockcol*TILE_WIDTH);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(ARYTYPE *Md, ARYTYPE *Nd, ARYTYPE *Pd, int width)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread block computes one sub-matrix Psub of P
    ARYTYPE *Pd_sub = GetSubMatrix(Pd, blockRow, blockCol, width);
    
    // Thread row and column within sub-matrix
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Each thread computes one element of Psub
    // by accumulating results into Pvalue
    ARYTYPE Pvalue = 0;
    
    // Shared memory used to store Msub and Nsub respectively
    __shared__ ARYTYPE Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ ARYTYPE Nds[TILE_WIDTH][TILE_WIDTH];
    
    // Loop over all the sub-matrices of M and N that are
    // required to compute Psub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (width / TILE_WIDTH); ++m) {
        // Get sub-matrix Msub of M
        ARYTYPE *Md_sub = GetSubMatrix(Md, blockRow, m, width);
        // Get sub-matrix Nsub of N
        ARYTYPE *Nd_sub = GetSubMatrix(Nd, m, blockCol, width);
        
        // Load Msub and Nsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        Mds[row][col] = GetElement(Md_sub, row, col, width);
        Nds[row][col] = GetElement(Nd_sub, row, col, width);
        
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Msub and Nsub together
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[row][k] * Nds[k][col];
            //Pvalue += __mul24(Mds[row][k],Nds[k][col]);
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of M and N in the next iteration
        __syncthreads();
    }
    
    // Write Psub to device memory
    // Each thread writes one element
    SetElement(Pd_sub, row, col, width, Pvalue);
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of TILE_WIDTH
void MatMul(ARYTYPE *M, ARYTYPE *N, ARYTYPE *P, int width)
{
    size_t size = width * width * sizeof(ARYTYPE);
    ARYTYPE *Md, *Nd, *Pd;
    
    // Allocate and Load M, N to device memory
    cudaMalloc((void **)&Md, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&Nd, size);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    
    // Allocate P on the device
    cudaMalloc((void **)&Pd, size);
    
    // Setup the execution configuration
    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    // Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Invoke kernel
    MatMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
    cudaError_t cuda_err = cudaGetLastError();
    if ( cudaSuccess != cuda_err ){
        printf("before kernel call: error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    
    // Get stop time event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Read P from device memory
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}
