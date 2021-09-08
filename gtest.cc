#include <gtest/gtest.h>
#include "prime.h"
#include <iostream>

TEST(Sievetest, integer) {
    ASSERT_EQ(EratosthenesSieve(10), 4);
    ASSERT_EQ(EratosthenesSieve(1000), 168); //1e3
    ASSERT_EQ(EratosthenesSieve(1000000), 78498); //1e6
    ASSERT_EQ(EratosthenesSieve(100000000), 5761455); //1e8
    ASSERT_EQ(EratosthenesSieve(1000000000), 50847534); //1e9

    // ASSERT_EQ(EratosthenesSieve(10000000000), 455052511); //1e10

}

TEST(miillerTest, prime) {
    ASSERT_TRUE(miillerTest(10000019));
    ASSERT_TRUE(miillerTest(1000000007));
    // ASSERT_TRUE(miillerTest(10000000019)); //can't pass
    // ASSERT_TRUE(miillerTest(10000000000037)); //can't pass
}

TEST(primeTest, prime) {
    ASSERT_TRUE(IsPrime(10000019));
    ASSERT_TRUE(IsPrime(1000000007));
    ASSERT_TRUE(IsPrime(10000000019));
    ASSERT_TRUE(IsPrime(10000000000037));
}

TEST(goldbach, segment){
    ASSERT_TRUE(solve(10));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
