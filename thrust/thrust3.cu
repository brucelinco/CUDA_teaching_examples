#include "book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main( void ) {

	thrust::host_vector<int> h_vec(2);

	thrust::device_vector<int> d_vec = h_vec;

	d_vec[0] = 13;
	d_vec[1] = 27;
	
	std::cout << "sum: " << d_vec[0] + d_vec[1] << std::endl;
 	return 0;
}







