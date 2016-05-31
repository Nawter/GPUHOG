#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "Global.h"
static cudaArray* arrayImageNormalize = NULL;
texture<uchar4, 2, cudaReadModeNormalizedFloat> textureImageNormalize;
extern "C++" __global__ void calculateGradientsDevice(int width, int height, int minX, int minY, int maxX,
                                       int maxY, int padX, int padY, float2* gradientDevice)
{
    float4	pixelUp;
    float4	pixelDown;
    float4	pixelLeft;
    float4	pixelRight;
    const float stepX = 1.f / (width-2*padX);
    const float stepY = 1.f / (height-2*padY);
    const float offX = stepX / 2.f;
    const float offY = stepY / 2.f;
    const int posX	= blockDim.x * blockIdx.x + threadIdx.x + minX;
    const int posY	= blockDim.y * blockIdx.y + threadIdx.y + minY;
    const int roiWidth  = maxX - minX + 2*padX-1;
    const int pixelIdx = (posX-minX) + (posY-minY) * roiWidth;
    if(	posX == 0 || posY == 0 || posX == maxX || posY == maxY )
    {
        gradientDevice[pixelIdx].x = 0.f;
        gradientDevice[pixelIdx].y = 0.f;
    }
    if(posX < maxX+2*padY && posX > 0 && posY < maxY+2*padY && posY > 0)
    {
        pixelDown  = tex2D(textureImageNormalize,(posX-padX) * stepX + offX,(posY+1-padY) * stepY + offY);
        pixelUp    = tex2D(textureImageNormalize,(posX-padX) * stepX + offX,(posY-1-padY) * stepY + offY);
        pixelLeft  = tex2D(textureImageNormalize,(posX-1-padX) * stepX + offX,(posY-padY) * stepY + offY);
        pixelRight = tex2D(textureImageNormalize,(posX+1-padX) * stepX + offX,(posY-padY) * stepY + offY);
#ifdef ENABLE_GAMMA_COMPRESSION
        pixelUp.x = sqrtf(	pixelUp.x);
        pixelUp.y = sqrtf(	pixelUp.y);
        pixelUp.z = sqrtf(	pixelUp.z);
        pixelUp.w = sqrtf(	pixelUp.w);
        pixelDown.x = sqrtf(pixelDown.x);
        pixelDown.y = sqrtf(pixelDown.y);
        pixelDown.z = sqrtf(pixelDown.z);
        pixelDown.w = sqrtf(pixelDown.w);
        pixelLeft.x = sqrtf(pixelLeft.x);
        pixelLeft.y = sqrtf(pixelLeft.y);
        pixelLeft.z = sqrtf(pixelLeft.z);
        pixelLeft.w = sqrtf(pixelLeft.w);
        pixelRight.x = sqrtf(pixelRight.x);
        pixelRight.y = sqrtf(pixelRight.y);
        pixelRight.z = sqrtf(pixelRight.z);
        pixelRight.w = sqrtf(pixelRight.w);
#endif
        float3 directionGradientDX, directionGradientDY;
        directionGradientDX.x = (pixelRight.x - pixelLeft.x);
        directionGradientDX.y = (pixelRight.y - pixelLeft.y);
        directionGradientDX.z = (pixelRight.z - pixelLeft.z);
        directionGradientDY.x = (pixelDown.x - pixelUp.x);
        directionGradientDY.y = (pixelDown.y - pixelUp.y);
        directionGradientDY.z = (pixelDown.z - pixelUp.z);
        float3 magnitudGradientDevice;
        magnitudGradientDevice.x = directionGradientDX.x * directionGradientDX.x + directionGradientDY.x * directionGradientDY.x;
        magnitudGradientDevice.y = directionGradientDX.y * directionGradientDX.y + directionGradientDY.y * directionGradientDY.y;
        magnitudGradientDevice.z = directionGradientDX.z * directionGradientDX.z + directionGradientDY.z * directionGradientDY.z;
        float direction;
        float magnitude;
        if(magnitudGradientDevice.z > magnitudGradientDevice.y)
        {
            if(magnitudGradientDevice.z > magnitudGradientDevice.x)
            {
                magnitude	= sqrtf(magnitudGradientDevice.z);
                direction	= atan2f(directionGradientDY.z, directionGradientDX.z) * 180.f / (float)M_PI;
            }
            else
            {
                magnitude	= sqrtf(magnitudGradientDevice.x);
                direction	= atan2f(directionGradientDY.x, directionGradientDX.x) * 180.f / (float)M_PI;
            }
        } else
        {
            if(magnitudGradientDevice.y > magnitudGradientDevice.x)
            {
                magnitude	= sqrtf(magnitudGradientDevice.y);
                direction	= atan2f(directionGradientDY.y, directionGradientDX.y) * 180.f / (float)M_PI;
            } else
            {
                magnitude	= sqrtf(magnitudGradientDevice.x);
                direction	= atan2f(directionGradientDY.x, directionGradientDX.x) * 180.f / (float)M_PI;
            }
        }
         gradientDevice[pixelIdx].x = direction;
         gradientDevice[pixelIdx].y = magnitude;
    }
}
extern "C++" __host__ int calculateGradientsHost(int padWidth, int padHeight, int minX, int minY, int maxX,
                                  int maxY, int padX, int padY, float2* gradientDevice)
{
    const int threadsX = 16;
    const int threadsY = 16;
   //const int threadsX = 4;
   //const int threadsY = 4;
    dim3 blocks(threadsX, threadsY);
    dim3 grid((int)ceil(padWidth/((float)threadsX)), (int)ceil(padHeight/((float)threadsY)) );
    calculateGradientsDevice<<< grid , blocks >>>(padWidth, padHeight,minX, minY, maxX, maxY,padX, padY, gradientDevice);
    MANAGE_ERRORS("ERROR:Failed to computeGradientsHost kernel\n");
    return 0;
}
int prepareImage(const unsigned char* imageHost, int width, int height)
{
    cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<uchar4>();
    //Allocates a CUDA array according to the cudaChannelFormatDesc structure
    // desc and returns a handle to the new CUDA array in *array.
    cudaMallocArray(&arrayImageNormalize, &channelDescriptor, width, height);
    MANAGE_ERRORS("ERROR:Failed to cuda malloc arrayImageNormalize in prepareImage\n");
    /*Copies count bytes from the memory area pointed to by src to the CUDA array dst starting at the upper left corner
    (wOffset, hOffset), where kind is one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
    or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
    cudaError_t cudaMemcpyToArray	(	struct cudaArray * 	dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * 	src,
    size_t 	count,
    enum cudaMemcpyKind 	kind
    )*/
    cudaMemcpyToArray(arrayImageNormalize, 0, 0, imageHost, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    MANAGE_ERRORS("ERROR:Failed to cuda memcpy to arrayImageNormalize in prepareImage\n");
    textureImageNormalize.addressMode[0] = cudaAddressModeClamp;
    textureImageNormalize.addressMode[1] = cudaAddressModeClamp;
    textureImageNormalize.filterMode = cudaFilterModeLinear;
    textureImageNormalize.normalized = true;
    //Texture memory only for read.
    //Binds the CUDA array array to the texture reference texref.
    //desc describes how the memory is interpreted when fetching values from the texture.
    //Any CUDA array previously bound to texref is unbound.
    cudaBindTextureToArray(textureImageNormalize, arrayImageNormalize, channelDescriptor);
    MANAGE_ERRORS("ERROR:Failed to bind textureImageNormalize in prepareImage\n");
    return 0;
}

int eraseImage()
{
    if(arrayImageNormalize)
    {
        cudaUnbindTexture(textureImageNormalize);
        MANAGE_ERRORS("ERROR:Failed to unbind textureImageNormalize in eraseImage");
        cudaFreeArray(arrayImageNormalize);
        MANAGE_ERRORS("ERROR:Failed to cuda free arrayImageNormalize in eraseImage");
        arrayImageNormalize = NULL;
    }
    return 0;
}
