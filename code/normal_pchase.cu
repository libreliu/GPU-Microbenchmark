#include <iostream>
#include <cuda_runtime.h>

#define CHK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define ONE_KILOBYTE 1024
#define ONE_MEGABYTE ((1024 * ONE_KILOBYTE))

// TODO: figure out optimization stuff
template<int stride, int elemCount, int loopCount>
__global__ void fixed_pchase(int *base)
{
    for (int loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        for (int i = 0; i < elemCount; i++) {
            base[stride * i] = i;
        }
    }
}

void bench_memory_sequential()
{
    int stride_max = 128 * ONE_MEGABYTE;
    int stride_min = 1;
    int arena_size = stride_max * 4;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *devMem;
    // TODO: test on flags
    cudaMallocManaged(&devMem, arena_size);

    CHK_CUDA(cudaEventRecord(start));
    fixed_pchase<1, 400000, 1000> <<<1, 1>>>(devMem);
    CHK_CUDA(cudaEventRecord(stop));

    // wait until event complete
    CHK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;

    std::cout << "Time elapsed: " << std::to_string(milliseconds) << std::endl;
}

int main() {
    CHK_CUDA(cudaSetDevice(0));

    bench_memory_sequential();

    return 0;
}
