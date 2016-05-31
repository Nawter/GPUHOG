#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include "Global.h"
#include <cuda.h>
cudaArray* gaussWeightsDevice = NULL;
cudaArray* bilinearWeightsDevice = NULL;
texture<float, 2, cudaReadModeElementType> textureGaussianWeights;
texture<float4, 2, cudaReadModeElementType> textureBilinearWeights;
extern "C++" __global__ void computeBlocksDevice(int width, int height, float2* gradientDevice, float* blocksDevice)
{
    volatile __shared__ float blocksDeviceShared[64][2*2][NBINS];		// 32 threads, 4 cells, 9 bins
    volatile __shared__ float squaresDeviceShared[4];
    const int threads =32;
    const int cells = 4;
    const int cellIdx = threadIdx.x;	// 0-3	cells in block
    const int columnIdx = threadIdx.y;	// 0-7	columns
    const int posIdX = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockX = (cellIdx % 2)*CELL_SIZE + columnIdx;
    const int blockY = cellIdx < 2 ? 0 : CELL_SIZE;
    const int pixelX = blockIdx.x * (BLOCK_WIDTH/2) + blockX;		// Assume 50% overlap
    const int pixelY = blockIdx.y * (BLOCK_HEIGHT/2) + blockY;
    for(int i=0; i < NBINS; i++)
    {
        for(int cell =0; cell < NUM_BLOCK_CELLS_X*NUM_BLOCK_CELLS_Y; cell++)
        {
            blocksDeviceShared[posIdX][cell][i] = 0.f;
        }
    }
    //first
    __syncthreads();
    if(pixelX < width && pixelY < height)
    {
        for(int i=0; i < CELL_SIZE; i++)
        {
            const int pixelIdx = (pixelY + i) * width + pixelX;
            float magnitude = gradientDevice[pixelIdx].y;
            float contribution = magnitude * tex2D(textureGaussianWeights, blockY+i, blockX);
            float binSize = 180.f / NBINS;
            float angle = gradientDevice[pixelIdx].x - binSize/2.f;
            if(angle < 0) angle += 180.f;
            float delta = (angle * NBINS) / 180.f;
            int leftBin = (int)floorf( delta );
            delta -= leftBin;
            int rightBin = leftBin >= NBINS-1 ? 0 : leftBin+1;
            if( leftBin < 0 ) leftBin = NBINS -1;
            float rightContribution = contribution * (delta);
            float leftContribution = contribution * (1-delta);
            float4 weights = tex2D(textureBilinearWeights, blockX, blockY+i);
            blocksDeviceShared[posIdX][0][leftBin] += leftContribution * weights.x;
            blocksDeviceShared[posIdX][0][rightBin]+= rightContribution * weights.x;
            blocksDeviceShared[posIdX][1][leftBin] += leftContribution * weights.y;
            blocksDeviceShared[posIdX][1][rightBin]+= rightContribution * weights.y;
            blocksDeviceShared[posIdX][2][leftBin] += leftContribution * weights.z;
            blocksDeviceShared[posIdX][2][rightBin]+= rightContribution * weights.z;
            blocksDeviceShared[posIdX][3][leftBin] += leftContribution * weights.w;
            blocksDeviceShared[posIdX][3][rightBin]+= rightContribution * weights.w;
        }
    }
    // Second
    __syncthreads();
      if(threadIdx.y == 0)
    {
        for(int i=1; i < 32; i++)
        {
            for(int bin=0; bin < NBINS; bin++)
            {
                blocksDeviceShared[0][threadIdx.x][bin] += blocksDeviceShared[i][threadIdx.x][bin];
            }
        }
    }
    // Third
    __syncthreads();
    const float epsilon = 0.036f * 0.036f;	// magic numbers
    const float eHys	= 0.1f * 0.1f;
    const float clipThreshold = 0.2f;
    if(threadIdx.y == 0 )
    {
        float ls = 0.f;
        for(int j=0; j < NBINS; j++)
        {
            ls += blocksDeviceShared[0][threadIdx.x][j] * blocksDeviceShared[0][threadIdx.x][j];
        }
        squaresDeviceShared[threadIdx.x] = ls;
    }
    // Fourth
    __syncthreads();
    if(threadIdx.y == 0 && threadIdx.x == 0 )
    {
        squaresDeviceShared[0] += squaresDeviceShared[1] + squaresDeviceShared[2] + squaresDeviceShared[3];
    }
    // Fifth
    __syncthreads();
    float normalization = rsqrtf(squaresDeviceShared[0]+epsilon);
    if(threadIdx.y == 0 )
    {
        for(int j=0; j < NBINS; j++)
        {
            blocksDeviceShared[0][threadIdx.x][j] *= normalization;
            blocksDeviceShared[0][threadIdx.x][j] = blocksDeviceShared[0][threadIdx.x][j] > clipThreshold ? clipThreshold : blocksDeviceShared[0][threadIdx.x][j];
        }
    }
    if(threadIdx.y == 0 )
    {
        float ls = 0.f;
        for(int j=0; j < NBINS; j++)
        {
            ls += blocksDeviceShared[0][threadIdx.x][j] * blocksDeviceShared[0][threadIdx.x][j];
        }
        squaresDeviceShared[threadIdx.x] = ls;
    }
    // Sixth
    __syncthreads();
    if(threadIdx.y == 0 && threadIdx.x == 0 )
    {
        squaresDeviceShared[0] += squaresDeviceShared[1] + squaresDeviceShared[2] + squaresDeviceShared[3];
    }
    normalization = rsqrtf(squaresDeviceShared[0]+eHys);
    if(threadIdx.y == 0 )
    {
        for(int j=0; j < NBINS; j++)
        {
            blocksDeviceShared[0][threadIdx.x][j] *= normalization;
        }
    }
    if(threadIdx.y == 0 )
    {
        const int lastIdx = NBINS*4 * (blockIdx.y * gridDim.x + blockIdx.x);
        for(int bin=0; bin < NBINS; bin++)
        {
            blocksDevice[lastIdx + threadIdx.x*NBINS + bin] = blocksDeviceShared[0][threadIdx.x][bin];
        }
    }
}
extern "C++" __host__ int prepareGaussWeights()
{
    const float gX = BLOCK_WIDTH / 2 - 0.5f;
    const float gY = BLOCK_HEIGHT / 2 - 0.5f;
    float gaussWeightsHost[BLOCK_WIDTH][BLOCK_HEIGHT];
    for(int j=0; j < BLOCK_HEIGHT; j++)
    {
        for(int i=0; i < BLOCK_WIDTH; i++)
        {
            gaussWeightsHost[i][j] = 1.f /(2.f * (float)M_PI * SIGMA) *
                    exp(- 0.5f * ( (i-gX)*(i-gX)/(SIGMA*SIGMA) + (j-gY)*(j-gY)/(SIGMA*SIGMA) ) );
        }
    }
    float sum = 0;
    // orden cambiado de los bucles con respecto al original
    for(int i=0; i < BLOCK_WIDTH; i++)
    {
        for(int j=0; j < BLOCK_HEIGHT; j++)
        {
            sum += gaussWeightsHost[i][j];
        }
    }
    for(int i=0; i < BLOCK_HEIGHT; i++)
    {
        for(int j=0; j < BLOCK_WIDTH; j++)
        {
            gaussWeightsHost[i][j] /= sum;
        }
    }
    cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float>();
    cudaMallocArray( &gaussWeightsDevice, &channelDescriptor, BLOCK_WIDTH, BLOCK_HEIGHT);
    MANAGE_ERRORS("ERROR:Failed to cuda malloc gaussWeightsDevice in prepareGaussWeights\n");
    cudaMemcpyToArray(gaussWeightsDevice, 0, 0, gaussWeightsHost,BLOCK_WIDTH * BLOCK_HEIGHT * sizeof(float),
                        cudaMemcpyHostToDevice);
    MANAGE_ERRORS("ERROR:Failed to cuda memcpy to gaussWeightsDevice in prepareGaussWeights\n");
    cudaBindTextureToArray( textureGaussianWeights, gaussWeightsDevice, channelDescriptor);
    MANAGE_ERRORS("ERROR:Failed to bind textureGaussianWeights in prepareGaussWeights\n");
    return 0;
}
extern "C++" __host__ int prepareBilinearWeights()
{
    float* bilinearWeightsHost = (float*)malloc(sizeof(float) * 4 * 2 * CELL_SIZE * 2 * CELL_SIZE);
    if(!bilinearWeightsHost)
    {
        std::cout<<"ERROR:Failed to malloc bilinearWeightsHost in prepareBilinearWeights\n";
        return -1;
    }
    float bilinearWeightsHostTop[CELL_SIZE*2];
    float bilinearWeightsHostBottom[CELL_SIZE*2];
    float bilinearWeightsHostLeft[CELL_SIZE*2];
    float bilinearWeightsHostRight[CELL_SIZE*2];
    memset(bilinearWeightsHostTop, 0, sizeof(float) * CELL_SIZE * 2);
    memset(bilinearWeightsHostBottom, 0, sizeof(float) * CELL_SIZE * 2);
    memset(bilinearWeightsHostLeft, 0, sizeof(float) * CELL_SIZE * 2);
    memset(bilinearWeightsHostRight, 0, sizeof(float) * CELL_SIZE * 2);
    int descriptor = 9;
    int k = 0;
    for(int i=0; i < 4; i++, k+=2)
    {
            bilinearWeightsHostLeft[i] = (descriptor+k) / 16.f;
            bilinearWeightsHostTop[i] = (descriptor+k) / 16.f;
    }
    descriptor = 15;
    k = 0;
    for(int i=4; i < 12; i++, k+=2)
    {
        bilinearWeightsHostLeft[i] = (descriptor-k) / 16.f;
        bilinearWeightsHostTop[i] = (descriptor-k) / 16.f;
    }
    for(int i=0; i < CELL_SIZE*2; i++)
    {
        bilinearWeightsHostRight[i] = bilinearWeightsHostLeft[CELL_SIZE*2-1-i];
        bilinearWeightsHostBottom[i] = bilinearWeightsHostTop[CELL_SIZE*2-1-i];
    }
    for(int i=0; i < 2*CELL_SIZE; i++)
    {
        for(int j=0; j < 2*CELL_SIZE; j++)
        {
            const int pos = 4 * (j * 2*CELL_SIZE + i);
            bilinearWeightsHost[pos+0] = bilinearWeightsHostLeft[i] * bilinearWeightsHostTop[j];
            bilinearWeightsHost[pos+1] = bilinearWeightsHostRight[i] * bilinearWeightsHostTop[j];
            bilinearWeightsHost[pos+2] = bilinearWeightsHostLeft[i] * bilinearWeightsHostBottom[j];
            bilinearWeightsHost[pos+3] = bilinearWeightsHostRight[i] * bilinearWeightsHostBottom[j];
        }
    }
    cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc<float4>();
    cudaMallocArray( &bilinearWeightsDevice, &channelDescriptor, 2*CELL_SIZE, 2*CELL_SIZE);
    MANAGE_ERRORS("ERROR:Failed to cuda malloc bilinearWeightsDevice in prepareBilinearWeights\n");
    cudaMemcpyToArray(bilinearWeightsDevice, 0, 0, bilinearWeightsHost,sizeof(float) * 4 * 2*CELL_SIZE * 2*CELL_SIZE,
                        cudaMemcpyHostToDevice);
    MANAGE_ERRORS("ERROR:Failed to cuda memcpy to bilinearWeightsDevice in prepareBilinearWeights\n");
    cudaBindTextureToArray( textureBilinearWeights, bilinearWeightsDevice, channelDescriptor);
    MANAGE_ERRORS("ERROR:Failed to bind textureBilinearWeights in prepareBilinearWeights\n");
    free(bilinearWeightsHost);
    return 0;
}
extern "C++" __host__ int finalizeComputeBlocks()
{
    cudaUnbindTexture(textureGaussianWeights);
    MANAGE_ERRORS("ERROR:Failed to unbind textureGaussianWeights in finalizeComputeBlocks\n");
    cudaFreeArray(gaussWeightsDevice);
    MANAGE_ERRORS("ERROR:Failed to cuda free gaussWeightsDevice in finalizeComputeBlocks\n");
    cudaUnbindTexture(textureBilinearWeights);
    MANAGE_ERRORS("ERROR:Failed to unbind textureBilinearWeights in finalizeComputeBlocks\n");
    cudaFreeArray(bilinearWeightsDevice);
    MANAGE_ERRORS("ERROR:Failed to cuda free bilinearWeightsDevice in finalizeComputeBlocks\n");
    return 0;
}
extern "C++" __host__ int computeBlocksHost(int width, int height, float2* gradientDevice,float* blocksDevice)
{
    const int threadsX = 8;
    const int threadsY = 8;
    //const int threadsX = 8;
    //const int threadsY = 16;
    dim3 blocks(threadsX,threadsY);
    dim3 grid;   
    grid.x = (int)floor((width) / 8.f);
    grid.y = (int)floor((height) / 8.f);
    computeBlocksDevice <<< grid , blocks >>> (width, height, gradientDevice, blocksDevice);
    MANAGE_ERRORS("ERROR:Failed to computeBlocksHost kernel");
    return 0;
}










