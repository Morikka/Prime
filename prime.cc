#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <cmath>
#include <vector>
#include <map>
#include "prime.h"

using namespace std;

const uint32_t L1D_CACHE_SIZE = 16384;
const uint32_t L = 100; //The segment length, equals to b - a, usually less than 1e8
const uint16_t K = 50; //The size of set Q, guess it will less than 65535?
const uint8_t A[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

vector<uint64_t> prime; //stores all prime numbers under limit
set<uint64_t> evennumber;
map<uint64_t, uint64_t> mp;

uint64_t EratosthenesSieve(uint64_t limit){
  uint32_t sq = sqrt(limit);
  uint32_t segment_size = max(sq, L1D_CACHE_SIZE);
  uint64_t count = (limit < 2) ? 0 : 1; //number of primes

  // we sieve primes >= 3
  uint32_t i = 3;
  uint32_t s = 3;
  uint64_t n = 3;

  vector<char> sieve(segment_size);
  vector<char> is_prime(sq + 1, true);
  vector<uint64_t> primes;
  vector<uint64_t> multiples;

  for (uint64_t low = 0; low <= limit; low += segment_size) {
    fill(sieve.begin(), sieve.end(), true);
    // current segment = [low, high]
    uint64_t high = low + segment_size - 1;
    high = min(high, limit);

    // generate sieving primes using simple sieve of Eratosthenes
    for (; (uint64_t) i * i <= high; i += 2)
      if (is_prime[i])
        for (uint64_t j = (uint64_t) i * i; j <= sq; j += i)
          is_prime[j] = false;
    // initialize sieving primes for segmented sieve
    for (; (uint64_t) s * s <= high; s += 2) {
      if (is_prime[s]) {
           primes.push_back(s);
        multiples.push_back(s * s - low);
      }
    }

    // sieve the current segment
    for (size_t i = 0; i < primes.size(); i++) {
      uint64_t j = multiples[i];
      for (int64_t k = primes[i] * 2; j < segment_size; j += k)
        sieve[j] = false;
      multiples[i] = j - segment_size;
    }

    for (; n <= high; n += 2){
      if (sieve[n - low]){ // n is a prime
        prime.push_back(n);          
        count++;  
      }
    }
  }
  return count;
}
 
// return x^y mod p
uint64_t power(uint64_t x, uint64_t y, uint64_t p){
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

// miller-rabin test for primality of n
// Smallest odd number for which Miller-Rabin primality test: http://oeis.org/A014233
bool miillerTest(uint64_t n){
    if (n <= 1 || n == 4)  return false;
    if (n <= 3) return true;

    uint64_t d = n - 1;
    uint64_t x;
    int8_t r = 0; // r is smaller than 64
    int8_t i; // i is smaller than 12
    int8_t j; // j is smaller than j

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
    // else if (n<318665857834031151167461) l = 12; //int64_t max = 9223372036854775807; 
    // else if (n<3317044064679887385961981) l = 13;
    for(i=0; i<l; i++){
        x = power(A[i], d, n);
        if(x == 1) continue;
        for(j=0; j<r; j++){
            if(x == n-1) break;
            x = (x * x) % n;
        }
    if(j >= r) return 0;
    }
    return 1;
}

void combine(uint64_t n, uint64_t a, uint64_t b){
  uint64_t tmpsum;
  for(size_t i=0; i<prime.size();i++){
    tmpsum = n + prime[i];
    if(tmpsum>a && tmpsum<=b){
      if(evennumber.find(tmpsum)==evennumber.end()){
        // cout<<tmpsum<<" "<<n<<" "<<prime[i]<<endl;
        mp[tmpsum] = prime[i];
        evennumber.insert(n+prime[i]);
      }
    }
  }
}

void find(uint64_t a, uint64_t b){
  int k = 0;
  int numcount = L/2;
  int64_t i;
  for(i=a-1; i>=a-L; i-=2){
    if(evennumber.size() >= numcount){
      break;
    }
    if(miillerTest(i)){
      combine(i, a, b);
      k++;
    }
  }
  cout<<"k and i are: "<<k<<" "<<i<<endl;
}

bool IsPrime(uint64_t n){
  if(n<2) return false;
  for(uint32_t i = 2; (uint64_t)i * i <= n; ++i){
    if(n%i == 0) return false;
  }
  return true;
}

bool result(uint64_t a, uint64_t b){
  for(uint64_t i=a+2; i<=b; i+=2){
    if(!IsPrime(i - mp[i])){
      // cout<<"This is not correct: "<<i<<" = "<<mp[i]<<" + "<< i - mp[i]<<endl;
      return false;
    }
  }
  cout<<"The results are correct"<<endl;
  return true;
}

bool solve(uint64_t a){
  clock_t start, stop;
  start = clock();
  uint64_t b = a + (uint64_t) L;
  EratosthenesSieve(2*L);
  find(a, b);
  stop = clock();
  float elapsedTime = (float)(stop - start) /  (float)CLOCKS_PER_SEC * 1000.0f;
  printf("Time to generate:  %3.1f ms\n", elapsedTime );
  return result(a, b);
}