// Computes number of primes under N using Sieve of Eratosthenes
// N smaller or equal to int_max
#include<cstdio>
#include<cstring>
#include<iostream>
#include<vector>

using namespace std;


const long long N = 100000;

long long is_prime[N+2];

int primesieve(long long n){
  int p = 0;
  vector<long long> prime;
  for(long long i=0; i<=n; i++){
    is_prime[i] = 1;
  }
  is_prime[0] = is_prime[1] = 0;
  for(long long i=2; i<=n; i++){
    if(is_prime[i]){
      prime.push_back(i);
    }
    if((long long)i*i<=n){ //Note this boundary
      for(long long j = i*i; j<=n; j+=i){
        is_prime[j] = 0;
      }
    }
  }
  for(int i=0; i<prime.size(); i++){
    cout<<prime[i]<<" ";
  }
  return prime.size();
}

int main(){
  freopen("sieve.out", "w", stdout);
  primesieve(N);
  return 0;
}