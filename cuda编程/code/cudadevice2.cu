cudaSetDevice(0); // Set device 0 as current

float *p0;
size_t size = 1024 * sizeof(float) ;
cudaMalloc(&p0, size); // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
cudaSetDevice(1); // Set device 1 as current
int canAccessPeer = 0;
cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0); // Check whether device 1 can access device 0
if (canAccessPeer) {
    cudaDeviceEnablePeerAccess(0, 0); // Enable peer-to-peer access with device 0
    // Launch kernel on device 1
    // This kernel launch can access memory on device 0 at address p0
    MyKernel<<<1000, 128>>>(p0);
}
