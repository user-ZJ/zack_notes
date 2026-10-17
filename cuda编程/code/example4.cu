// Host code
int width = 64, height = 64;

float *devPtr;
size_t pitch;
// width参数以字节为单位，返回的pitch是每行实际占用的字节数（含对齐填充）
cudaMallocPitch((void **)&devPtr, &pitch, width*sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);
// Device code
__global__ void MyKernel(float *devPtr,size_t pitch, int width, int height){
    for ( int r = 0; r < height; ++r){
        float *row = (float *)((char *)devPtr + r*pitch);
        for ( int c = 0; c <width; ++c) {
            float element = row[c];
        }
    }
}