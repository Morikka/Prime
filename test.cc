#include <iostream>
using namespace std;
 
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
 
bool miillerTest(int n){

    if (n <= 1 || n == 4)  return false;
    if (n <= 3) return true;

    int d = n - 1;
    int r = 0;
    int a[] = {2, 3};
    int64_t x;
    int i,j;

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

int main() {
    cout<<miillerTest(19940909)<<endl;
    // freopen("t1.out","w",stdout);
    // for (int n = 1; n <= 100000; n++)
    //    if (miillerTest(n))
    //       cout << n <<endl;
    return 0;
}