#include "book.h"
#include <time.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

	//perform sum on host (CPU)

	thrust::transform(h_a.begin(), h_a.end(), h_b.begin(), h_c.begin(), thrust::plus<int>());
	thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(), thrust::plus<int>());

	
	int i;
	int pass = 1;
	for(i=0; i<16; i++){
	   if(h_c[i]!= d_c[i]){
		cout<<"error: "<<h_c[i]<< "!=" << d_c[i] << endl;
		pass = 0;
	   }
	}
	if(pass==1)
	   cout<<"Succeed"<<endl;


 	return 0;
}







