#include "book.h"
#include <time.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace std;

int main( void ) {


	thrust::host_vector<int> h_a( 16 );
	thrust::host_vector<int> h_b( 16 );
	thrust::host_vector<int> h_c( 16 );

	thrust::generate(h_a.begin(), h_a.end(), rand);
	thrust::generate(h_b.begin(), h_b.end(), rand);

	// transfer data to the device
	thrust::device_vector<int> d_a = h_a;
	thrust::device_vector<int> d_b = h_b;
	thrust::device_vector<int> d_c(16);



	thrust::transform(h_a.begin(), h_a.end(), h_b.begin(), h_c.begin(), thrust::multiplies<int>());
 	int h_dot = thrust::reduce(h_c.begin(), h_c.end());

	thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(), thrust::multiplies<int>());
	//int d_dot = thrust::reduce(d_c.begin(), d_c.end());
	int d_dot = thrust::reduce(d_c.begin(), d_c.end(), 0, thrust::plus<int>());
	

	if(h_dot!= d_dot){
	   cout<<"error: "<<h_dot<< "!=" << d_dot << endl;
	}
	else cout<<"Succeed"<<endl;


 	return 0;
}







