void CUDART_CB MyCallback(void *data){
    printf ("Inside callback %d\n", (int)data);
}

for( int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size,cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i],size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size,cudaMemcpyDeviceToHost, stream[i]);
    cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0);
}