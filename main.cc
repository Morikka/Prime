#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <cmath>
#include <vector>
#include <map>

using namespace std;

const int64_t L1D_CACHE_SIZE = 32;
const int64_t L = 10000000;
const int64_t K = 50;

vector<int64_t> prime;
set<int64_t> evennumber;
map<int64_t, int64_t> mp;

void EratosthenesSieve(int64_t limit){
  int64_t sqrt = (int64_t) std::sqrt(limit);
  int64_t segment_size = std::max(sqrt, L1D_CACHE_SIZE);
  int64_t count = (limit < 2) ? 0 : 1;

  // we sieve primes >= 3
  int64_t i = 3;
  int64_t n = 3;
  int64_t s = 3;

  vector<char> sieve(segment_size);
  vector<char> is_prime(sqrt + 1, true);
  vector<int64_t> primes;
  vector<int64_t> multiples;

  for (int64_t low = 0; low <= limit; low += segment_size) {
    fill(sieve.begin(), sieve.end(), true);
    // current segment = [low, high]
    int64_t high = low + segment_size - 1;
    high = std::min(high, limit);

    // generate sieving primes using simple sieve of Eratosthenes
    for (; i * i <= high; i += 2)
      if (is_prime[i])
        for (int64_t j = i * i; j <= sqrt; j += i)
          is_prime[j] = false;
    // initialize sieving primes for segmented sieve
    for (; s * s <= high; s += 2) {
      if (is_prime[s]) {
           primes.push_back(s);
        multiples.push_back(s * s - low);
      }
    }

    // sieve the current segment
    for (std::size_t i = 0; i < primes.size(); i++) {
      int64_t j = multiples[i];
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
}
 
int64_t power(int64_t x, unsigned int y, int p){
    int64_t res = 1;
    x = x % p;
    while (y > 0){
        if (y & 1)
            res = (res*x) % p;
        y = y>>1;
        x = (x*x) % p;
    }
    return res;
}
 
bool miillerTest(int64_t n){
    if (n <= 1 || n == 4)  return false;
    if (n <= 3) return true;

    int64_t d = n - 1;
    int64_t r = 0;
    int64_t a[] = {2, 3};
    int64_t x;
    int64_t i,j;

    while (d % 2 == 0){
        d /= 2; 
        r++;
    }

    for(i=0; i<2; i++){
        x = power(a[i], d, n);
        if(x == 1) continue;
        for(j=0; j<r; j++){
            if(x == n-1) break;
            x = (x * x) % n;
        }
    if(j >= r) return 0;
    }
    return 1;
}

void combine(int64_t n, int64_t a, int64_t b){
  int64_t tmpsum;
  for(int i=0; i<prime.size();i++){
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

void find(int64_t a, int64_t b){
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
  cout<<"k and i is: "<<k<<" "<<i<<endl;
}

bool judge(int64_t n){
  if(n<2) return false;
  for(int64_t i = 2; i * i <= n; ++i){
    if(n%i == 0) return false;
  }
  return true;
}

void result(int64_t a, int64_t b){
  for(int64_t i=a+2; i<=b; i+=2){
    if(!judge(i - mp[i])){
      cout<<"This is not correct: "<<i<<" = "<<mp[i]<<" + "<< i - mp[i]<<endl;
      return;
    }
  }
  cout<<"The results are correct"<<endl;
}

int main(){
  freopen("data.out", "w", stdout);
  int64_t a, b;
  a = 10000000000;
  b = a + L;
  clock_t start, stop;
  start = clock();
  EratosthenesSieve(2*L);
  find(a, b);
  stop = clock();
  float elapsedTime = (float)(stop - start) /  (float)CLOCKS_PER_SEC * 1000.0f;
  printf( "Time to generate:  %3.1f ms\n", elapsedTime );
  result(a, b);
  return 0;
}