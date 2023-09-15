cudaEvent t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
for ( int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i*size, inputHost + i*size,
                        size , cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDev + i*size, inputDev + i*size, size ) ;
    cudaMemcpyAsync(outputHost + i*size, outputDev + i*size,
        size , cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);