#include "cuda.h"
#include "book.h"
#include<iostream>

#define PRIMECOUNT  664579
#define S 1850000
#define BEGIN 2000000
#define END 3100000

__device__ uint64_t power(uint64_t x, uint64_t y, uint64_t p){
    uint64_t res = 1;
    x = x % p;
    while (y > 0){
        if (y & 1)
            res = (res*x) % p;
        y = y>>1;
        x = (x*x) % p;
    }
    return res;
}

__device__ bool isPrime(uint64_t n){
    if(n<=2 || n==4) return false;
    if(n==3) return true;
    uint64_t d = n - 1;
    uint64_t x;
    int8_t r = 0; // r is smaller than 64
    int8_t i; // i is smaller than 12
    int8_t j; // j is smaller than j

    const uint8_t A[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

    while (d % 2 == 0){
        d /= 2; 
        r++;
    }
    
    int8_t l;
    if(n<2047) l = 1;
    else if (n<1373653) l = 2;
    else if (n<25326001) l = 3;
    else if (n<3215031751) l = 4;
    else if (n<2152302898747) l = 5;
    else if (n<3474749660383) l = 6;
    else if (n<341550071728321) l = 7;
    else if (n<3825123056546413051) l = 9;
    else l = 12;

    for(i=0; i<l; i++){
        x = power(A[i], d, n);
        if(x == 1) continue;
        for(j=0; j<r; j++){
            if(x == n-1) break;
            x = (x * x) % n;
        }
    if(j >= r) return false;
    }
    return true;
}

__device__ int binary_search(int *prime, int val){
    int l = 0;
    int r = PRIMECOUNT - 1;
    int ans = r, m;
    while(l<=r){
        m = (l+r)/2;
        if(prime[m]>=val){
            ans = m;
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return ans;
}

__global__ void goldbach(int *prime, int* sum){
    uint64_t tid = (threadIdx.x + blockIdx.x * blockDim.x) * 2 + 1 + S;
    bool flag = isPrime(tid);
    if(!flag) return;
    // printf("Prime: %d\n", tid);

    int idx;
    int tmp;
    int l = binary_search(prime, BEGIN - tid);
    int r = binary_search(prime, END + 1 - tid);

    // printf("%d %d\n", l, r);
    for(int i=l; i<r; i++){
        tmp = tid + prime[i];
        idx = (tmp - BEGIN) / 2;

        if(idx<0||idx>(END-BEGIN)/2){
            printf("%d\n", idx);
            continue;
        }

        sum[idx] = tid;
    }
    return;
}

int main( void ) {
    // Read small prime numbers
    freopen("sieve.out", "r", stdin);
    int prime[PRIMECOUNT];
    for(int i=0; i<PRIMECOUNT; i++)
        std::cin>>prime[i];

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    // HANDLE_ERROR(cudaEventRecord(start,0));
    int *sum;
    int *dev_sum;
    int *dev_prime;

    int L = (END-BEGIN)/2 + 1;
    sum = (int*)malloc(L * sizeof(int));

    HANDLE_ERROR(cudaMalloc((void**)&dev_sum, L*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_prime, PRIMECOUNT * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_prime, &prime, PRIMECOUNT * sizeof(int), 
                            cudaMemcpyHostToDevice));

    goldbach<<<1024,1024>>>(dev_prime, dev_sum);
    // HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(sum, dev_sum, L * sizeof(int), 
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_sum));

    // export data here
    for(int i=0; i<L; i++){
        std::cout<< i * 2 + BEGIN <<" = " <<sum[i] <<" + "<<i * 2 + BEGIN - sum[i]<<std::endl;
    }

    free(sum);

    return 0;
}