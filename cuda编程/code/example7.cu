// cudaLaunchHostFunc的回调签名为 void(void *userData)
// 回调中不能调用任何CUDA API
void CUDART_CB MyCallback(void *data){
    printf("Inside callback %d\n", (int)(size_t)data);
}

for( int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size,cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i],size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size,cudaMemcpyDeviceToHost, stream[i]);
    cudaLaunchHostFunc(stream[i], MyCallback, (void*)(size_t)i);
}

// 旧接口cudaStreamAddCallback的回调签名不同，flags为保留参数，必须为0：
// void CUDART_CB MyStreamCallback(cudaStream_t stream, cudaError_t status, void *data);
// cudaStreamAddCallback(stream[i], MyStreamCallback, (void*)(size_t)i, 0);
