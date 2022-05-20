#include <iostream>

__device__ int x;

__global__ void test_kernel(void)
{
    x = 42;
}

static void run_test(void)
{
    std::cout << "Running unaligned_kernel: ";
    test_kernel<<<1,1>>>();
    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
}

int main() {

    std::cout << "Mallocing memory" << std::endl;

    run_test();

    return 0;
}
