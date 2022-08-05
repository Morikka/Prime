# Prime

A mini CUDA project used to test binary Goldbach conjecture on [a, b].

## Introduction

Goldbach conjecture is a famous mathematical problem which has not been confirmed by now.

## Compile

### Compile the non-cuda version

Using gtest:

```bash
g++ -std=c++17 -o gtest gtest.cc prime.cc -lgtest -lpthread
./gtest
```

Expected output

```bash
[==========] Running 4 tests from 4 test suites.
[----------] Global test environment set-up.
[----------] 1 test from Sievetest
[ RUN      ] Sievetest.integer
[       OK ] Sievetest.integer (7009 ms)
[----------] 1 test from Sievetest (7009 ms total)

[----------] 1 test from miillerTest
[ RUN      ] miillerTest.prime
[       OK ] miillerTest.prime (0 ms)
[----------] 1 test from miillerTest (0 ms total)

[----------] 1 test from primeTest
[ RUN      ] primeTest.prime
[       OK ] primeTest.prime (18 ms)
[----------] 1 test from primeTest (18 ms total)

[----------] 1 test from goldbach
[ RUN      ] goldbach.segment
k and i are: 12 917
Time to generate:  2212.5 ms
The results are correct
[       OK ] goldbach.segment (2212 ms)
[----------] 1 test from goldbach (2212 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 4 test suites ran. (9241 ms total)
[  PASSED  ] 4 tests.
```

### Compile the cuda (parallel) version

```bash
nvcc -o prime prime.cu
./prime
```

## Reference

[1]: Deshouillers, J-M., Herman JJ te Riele, and Yannick Saouter. "New experimental results concerning the Goldbach conjecture." International Algorithmic Number Theory Symposium. Springer, Berlin, Heidelberg, 1998.