#include "book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#define N 32


int main( void ) {

	// generate 16M random numbers on the host
	thrust::host_vector<int> h_vec( 16*1024*1024 );
	thrust::generate(h_vec.begin(), h_vec.end(), rand);
	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;
	thrust::sort(d_vec.begin(), d_vec.end()); // sort data on the device
	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

 	return 0;
}







