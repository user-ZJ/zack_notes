cudaStream t stream[2];
for ( int i = 0; i < 2; ++i)
cudaStreamCreate(&stream[i]);

float *hostPtr;
cudaMallocHost(&hostPtr, 2*size);
for ( int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i*size, hostPtr + i*size,size , 
                                cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>(outputDevPtr + i*size, inputDevPtr + i*size, size);
    cudaMemcpyAsync(hostPtr + i*size, outputDevPtr + i*size,size ,
                                cudaMemcpyDeviceToHost, stream[i]);
}
for ( int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);