/**
 * This checks for how many SMs,
 * threads, blocks, etc. there are
 * on a given Nvidia GPU
 * Can also be grabbed in program using:
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // <-- this is the function that does everything
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Shared mem per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Warp size: %d\n", prop.warpSize);
    return 0;
}