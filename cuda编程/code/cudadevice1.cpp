#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;

int main()
{
    int device_count;
    cudaGetDeviceCount(&device_count); // GPU个数
    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "totalGlobalMem: " << prop.totalGlobalMem / 1024.0 / 1024 << "MB" << std::endl;
        // computeMode：设备计算模式。
        // cudaMemGetInfo查询的是当前设备，需要先切换到设备i
        cudaSetDevice(i);
        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        std::cout << "Total memory: " << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Free memory: " << free_byte / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "used memory: " << (total_byte-free_byte) / (1024.0 * 1024.0) << " MB" << std::endl;
    }
    return 0;
}