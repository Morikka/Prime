#include "cuda.h"


#define S 1500
#define A 2000
#define B 3000

__global bool isPrime(int64_t n){
    if(n<=2 || n==4) return false;
    if(n==3) return true;

}
__global__ void prime(){
    int64_t tid = threadIdx.x + blockIdx.x * blockDim.x + S;

    printf("%d\n", tid);
    int idx = (tid - A)/2;
}

int main( void ) {
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    // HANDLE_ERROR(cudaEventRecord(start,0));

    int *sum;
    int *dev_sum;
    int L = (B-A)/2;
    sum = (int*)malloc(L * sizeof(int));
    HANDLE_ERROR(cudaMalloc((void**)&dev_sum, L*sizeof(int)));

    prime<<<16,16>>>();

    HANDLE_ERROR(cudaMemcpy(sum, dev_sum, L * sizeof(int), 
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_sum));
    free(sum);

    return 0;
}