#include<stdio.h>
#include"scrImagePgmPpmPackage.h"

//Kernel which calculate the resized image
__global__ void createResizedImage(unsigned char *imageScaledData, int scaled_width, int scaled_height,
                                   float scale_factor, cudaTextureObject_t texObj)
{
	const unsigned int tidX = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int tidY = blockIdx.y*blockDim.y + threadIdx.y;
	if (tidX >= scaled_width || tidY >= scaled_height)
		return;
	const unsigned index = tidY*scaled_width+tidX;
       	
	// Step 4: Read the texture memory from your texture reference in CUDA Kernel
	imageScaledData[index] = tex2D<unsigned char>(texObj,(float)(tidX*scale_factor),(float)(tidY*scale_factor));
}

int main(int argc, char*argv[])
{
	int height=0, width =0, scaled_height=0,scaled_width=0;
	//Define the scaling ratio	
	float scaling_ratio=0.5;
	unsigned char*data;
	unsigned char*scaled_data,*d_scaled_data;

	char inputStr[1024] = {"aerosmith-double.pgm"};
	char outputStr[1024] = {"aerosmith-double-scaled.pgm"};
	cudaError_t returnValue;

	//Create a channel Description to be used while linking to the tecture
	cudaArray* cu_array = NULL;
	cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);

	get_PgmPpmParams(inputStr, &height, &width);	//getting height and width of the current image
	data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	printf("\n Reading image width height and width [%d][%d]", height, width);
	scr_read_pgm( inputStr , data, height, width );//loading an image to "inputimage"

	scaled_height = (int)(height*scaling_ratio);
	scaled_width = (int)(width*scaling_ratio);
	scaled_data = (unsigned char*)malloc(scaled_height*scaled_width*sizeof(unsigned char));
	printf("\n scaled image width height and width [%d][%d]", scaled_height, scaled_width);

	//Allocate CUDA Array
	returnValue = cudaMallocArray( &cu_array, &channelDesc, width, height);
	if(returnValue == cudaSuccess)
		// CUDA数组不是线性内存，需要用cudaMemcpy2DToArray拷贝，不能用cudaMemcpy
		returnValue = cudaMemcpy2DToArray(cu_array, 0, 0, data, width * sizeof(unsigned char),
		               width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

	if(returnValue != cudaSuccess)
		printf("\n Got error while running CUDA API Array Copy: %s", cudaGetErrorString(returnValue));

	// Step 1. Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cu_array;
	// Step 2. Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Step 3: Create texture object
	cudaTextureObject_t texObj = 0;
	returnValue = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	if(returnValue != cudaSuccess) 
		printf("\n Got error while creating texture object: %s", cudaGetErrorString(returnValue));
	
	d_scaled_data = NULL;
	cudaMalloc(&d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char) );

	dim3 dimBlock(32, 32,1);
	// 向上取整，保证图像尺寸不能被32整除时也能覆盖所有像素
	dim3 dimGrid((scaled_width + dimBlock.x - 1)/dimBlock.x, (scaled_height + dimBlock.y - 1)/dimBlock.y, 1);
	printf("\n Launching grid with blocks [%d][%d] ", dimGrid.x,dimGrid.y);

	createResizedImage<<<dimGrid, dimBlock>>>(d_scaled_data,scaled_width,scaled_height,1/scaling_ratio, texObj);

	returnValue = cudaGetLastError();
	if(returnValue == cudaSuccess)
		returnValue = cudaDeviceSynchronize();
	if(returnValue == cudaSuccess)
		returnValue = cudaMemcpy(scaled_data, d_scaled_data,
		              scaled_height*scaled_width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if(returnValue != cudaSuccess) 
		printf("\n Got error while running CUDA API kernel: %s", cudaGetErrorString(returnValue));

	// Step 5: Destroy texture object
	cudaDestroyTextureObject(texObj);
	//storing the image with the detections
	scr_write_pgm( outputStr, scaled_data, scaled_height, scaled_width, "####" ); 
		
	if(data != NULL)
		free(data);
	if(cu_array !=NULL)
		cudaFreeArray(cu_array);
	if(scaled_data != NULL)
		free(scaled_data);
	if(d_scaled_data!=NULL)
		cudaFree(d_scaled_data);
	
	return 0;
}
