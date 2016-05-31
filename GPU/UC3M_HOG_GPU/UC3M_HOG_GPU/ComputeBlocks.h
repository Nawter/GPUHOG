#ifndef __COMPUTE_BLOCKS_H__
#define __COMPUTE_BLOCKS_H__
extern "C++" __global__ void computeBlocksDevice(int width, int height, float2* gradientDevice, float* blocksDevice);
extern "C++" __host__ int prepareGaussWeights();
extern "C++" __host__ int prepareBilinearWeights();
extern "C++" __host__ int finalizeComputeBlocks();
extern "C++" __host__ int computeBlocksHost(int width, int height, float2* gradientDevice,float* blocksDevice);
#endif

