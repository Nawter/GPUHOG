#ifndef __CALCULATE_GRADIENTS_H__
#define __CALCULATE_GRADIENTS_H__
extern "C++" __global__ void calculateGradientsDevice(int width, int height, int minX, int minY, int maxX,
                                       int maxY, int padX, int padY, float2* gradientDevice);
extern "C++" __host__ int calculateGradientsHost(int padWidth, int padHeight, int minX, int minY, int maxX,
                                         int maxY, int padX, int padY, float2* gradientDevice);
int prepareImage(const unsigned char* imageHost, int width, int height);
int eraseImage();
#endif
