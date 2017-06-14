#include<opencv/cv.h>
#include<opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define VALUE_MAX 10000000.0
#define iABS(x) (((x)<0)?(-(x)):(x))
#define iAbsDiff(a,b) (((a)<(b))?((b)-(a)):((a)-(b)))
	
struct match{
	int bestRow;
	int bestCol;
	int bestSAD;
}position;


const int threadsPerBlock = 16*16;

__global__ void kernel (unsigned char* sourcePtr, unsigned char* patternPtr, int* resultPtr, int s_height, int s_width, int p_height, int p_width);

void device_call(char* sourceImgPtr, char* patternImgPtr, int* host_result, int s_h, int s_w, int p_h, int p_w, int r_h, int r_w){
	//allocate the momory on GPU	
	unsigned char *dev_sourceImg, *dev_patternImg; 
	int *dev_result;
	cudaMalloc((void**)&dev_sourceImg, s_h * s_w * sizeof(char));	
	cudaMalloc((void**)&dev_patternImg, p_h * p_w * sizeof(char));	
	cudaMalloc((void**)&dev_result, r_h * r_w * sizeof(int));	
	
		// Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	//copy source and pattern image to GPU
	
	cudaMemcpy(dev_sourceImg, 
	           sourceImgPtr, 
			   s_h * s_w * sizeof(char),
			   cudaMemcpyHostToDevice);
			   
	cudaMemcpy(dev_patternImg, 
			   patternImgPtr, 
			   p_h * p_w * sizeof(char),
			   cudaMemcpyHostToDevice);
	
	dim3 grids(r_w, r_h);//each block handles a position 
	dim3 threads(16, 16);
	
	kernel<<<grids, threads>>>(dev_sourceImg, dev_patternImg, dev_result, s_h, s_w, p_h, p_w);
	
	cudaMemcpy(host_result, 
	           dev_result, 
			   r_h * r_w * sizeof(int),
			   cudaMemcpyDeviceToHost);
	// Get stop time event    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
	// Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}



/*new kernel function*/
__global__ void kernel (unsigned char* sourcePtr, unsigned char* patternPtr, int* resultPtr, int s_height, int s_width, int p_height, int p_width){
	int s_x = blockIdx.x + threadIdx.x;
	int s_y = blockIdx.y + threadIdx.y;
	int p_x = threadIdx.x; 
	int p_y = threadIdx.y;	
	int cacheIndex = p_y * blockDim.x + p_x;
	int r_height = s_height-p_height+1;
	int r_width = s_width-p_width+1;
	
	__shared__ int imgDIFF [threadsPerBlock];//16*16=256 elements
	
    // calculate image difference of source image and pattern image beginning 
	// at the position x and y of the source image
	int diff = 0;
	while(p_y < p_height){
		diff += iABS(sourcePtr[s_y*s_width + s_x]-patternPtr[p_y*p_width + p_x]);
		s_x += blockDim.x;
		p_x += blockDim.x;
		if(p_x >= p_width){
			s_x = blockIdx.x + threadIdx.x;
			p_x = threadIdx.x;
			s_y += blockDim.y;
			p_y += blockDim.y;
		}
	}
	imgDIFF[cacheIndex]=diff; 
	__syncthreads(); 
	 
	int i = threadsPerBlock/2;
    while (i != 0) {
        if (cacheIndex < i)
            imgDIFF[cacheIndex] += imgDIFF[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        resultPtr[blockIdx.y*r_width+blockIdx.x] = imgDIFF[0];
}

int main( int argc, char** argv )
{

	IplImage* sourceImg; 
	IplImage* patternImg; 
	int minSAD = VALUE_MAX;
	int SAD;
	int x, y, i, j;
	uchar* ptr;
	uchar p_sourceIMG, p_patternIMG;
	CvPoint pt1, pt2;
	int *host_result;
	int result_height, result_width;
	IplImage* sourceImgGuss;
    IplImage* patternImgGuss;


		
	if( argc != 3 )
	{
	    printf("Using command: %s source_image search_image\n",argv[0]);
		exit(1);
	}
	if((sourceImg = cvLoadImage( argv[1], 0)) == NULL){
		printf("%s cannot be openned\n",argv[1]);
		exit(1);
	}
	printf("height of sourceImg:%d\n",sourceImg->height);
	printf("width of sourceImg:%d\n",sourceImg->width);
	printf("size of sourceImg:%d\n",sourceImg->imageSize);
	
	
	if((patternImg = cvLoadImage( argv[2], 0)) == NULL){
		printf("%s cannot be openned\n",argv[2]);
		exit(1);
	}    
	printf("height of sourceImg:%d\n",patternImg->height);
	printf("width of sourceImg:%d\n",patternImg->width);
	printf("size of sourceImg:%d\n",patternImg->imageSize);
	
	//allocate memory on CPU to store SAD results
	result_height = sourceImg->height - patternImg->height + 1;
	result_width = sourceImg->width - patternImg->width + 1;
	host_result=(int*)malloc(result_height * result_width * sizeof(int));
	

	
	//call GPU.cu
	device_call(sourceImg->imageData, patternImg->imageData, host_result, sourceImg->height, sourceImg->width, patternImg->height, patternImg->width, result_height, result_width);
	
	
	for( y=0; y < result_height; y++ ) {
		for( x=0; x < result_width; x++ ) {
			if ( minSAD > host_result[y * result_width + x] ) {
				minSAD =  host_result[y * result_width + x];
				// give me VALUE_MAX
				position.bestRow = y;
				position.bestCol = x;
				position.bestSAD =  host_result[y * result_width + x];
			}
			
		}
	}
	

	printf("minSAD is %d\n", minSAD);
    
	//setup the two points for the best match
    pt1.x = position.bestCol;
    pt1.y = position.bestRow;
    pt2.x = pt1.x + patternImg->width;
    pt2.y = pt1.y + patternImg->height;

    // Draw the rectangle in the source image
    cvRectangle( sourceImg, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
			

	cvNamedWindow( "sourceImage", 1 );
    cvShowImage( "sourceImage", sourceImg );
	cvNamedWindow( "patternImage", 1 );
    cvShowImage( "patternImage", patternImg );
	
		
    cvWaitKey(0); 
 
    cvDestroyWindow( "sourceImage" );
    cvReleaseImage( &sourceImg );
	cvDestroyWindow( "patternImage" );
    cvReleaseImage( &patternImg );
    return 0;

}





