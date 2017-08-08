#include "book.h"
#include <time.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

int main( void ) {


	// generate 16M random numbers on the host
	thrust::host_vector<int> h_vec( 16*1024*1024 );
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;

	//perform sum on host (CPU)
        struct timespec t_start, t_end;
        double elapsedTime;

	clock_gettime( CLOCK_REALTIME, &t_start);
	int h_sum = thrust::reduce(h_vec.begin(), h_vec.end());
	clock_gettime( CLOCK_REALTIME, &t_end);

	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    	printf("elapsedTime on CPU: %lf ms\n", elapsedTime);

	// compute sum on device (GPU)
	
	clock_gettime( CLOCK_REALTIME, &t_start);
	int d_sum = thrust::reduce(d_vec.begin(), d_vec.end());

	clock_gettime( CLOCK_REALTIME, &t_end);
	
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;

    	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    	printf("elapsedTime on GPU: %lf ms\n", elapsedTime);

	cout << h_sum << " = " << d_sum << endl;


 	return 0;
}







