#include <iostream>
#include <fstream>
#include <cuda.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <vector>
#include "Global.h"
#include "CalculateGradients.h"
#include "TestPadImage.h"
#include "UseConversions.h"
#include "ComputeBlocks.h"
#include "TrainSVM.h"
#include "DumpDetections.h"
#include "SkipSupperssion.h"
#include "Timer.h"
#include "HOGPlanner.h"
#include "GlobalParameters.h"
#include "LocalParameters.h"
#include "HOG.h"
namespace UC3M_HOG_GPU
{
    double gradients, blocks, svmNum, end, nms;
    double totalGradients, totalBlocks, totalSvm, totalEnd;
    float*  blocksDevice = NULL;
    float2* gradientsDevice = NULL;
    int HOG::initialize()
    {
        int device = 0;
        if(cudaGetDeviceCount( &device) )
        {
           std::cout << "ERROR:cudaGetDeviceCount failed in HOG::initialize().\n";
           std::cout << "ERROR:CUDA driver and runtime version may be mismatched!\n";
        }
        if( device == 0 )
        {
           std::cout << "ERROR:no CUDA device available in HOG::initialize().\n";
           return -1;
        }
        int dev;
        for (dev = 0; dev < device; ++dev)
        {
            cudaDeviceProp deviceProperty;
            cudaGetDeviceProperties(&deviceProperty, dev);
            if (dev == 0)
            {
                if (deviceProperty.major == 9999 && deviceProperty.minor == 9999)
                {
                    std::cout << "ERROR:There is no device supporting CUDA in HOG::initialize().\n";
                }
                else if (device == 1)
                {
                    std::cout << "There is one device supporting CUDA.\n";
                }
                else
                {
                    std::cout << "There are " << device <<" devices supporting CUDA.\n";
                }
            }
            std::cout << "\nDevice "<< dev <<":" << "\""<< deviceProperty.name <<"\n";
        }
        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        cudaSetDevice(0);
        cudaDeviceReset();
        cudaDeviceReset();
        size_t f, t;
        std::cout << "driver: " << driverVersion / 1000 << "runtime:" << runtimeVersion / 1000 << "\n";
        cudaMemGetInfo(&f, &t);
        std::cout <<"before cudaMalloc " <<"free:" << f/1024*1024 << " bytes total:" <<t/1024*1024<<" bytes"<<"\n";
        //Weights
        prepareGaussWeights();
        prepareBilinearWeights();
        //Gradients
        cudaMalloc((void**)&gradientsDevice, MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION * sizeof(float2));
        MANAGE_ERRORS("ERROR:Failed to cuda malloc gradientDevice in HOG::initialize\n");
        cudaMemset(gradientsDevice, 0, sizeof(float2) * MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION);
        MANAGE_ERRORS("ERROR:Failed to cuda memset gradientDevice in HOG::initialize\n");
        //Blocks
        const int numBlocks = MAX_IMAGE_DIMENSION/8 * MAX_IMAGE_DIMENSION/8 ;
        const int blocksMemorySize = numBlocks * NUM_BLOCK_CELLS_X * NUM_BLOCK_CELLS_Y * NBINS * sizeof(float);
        cudaMalloc((void**)&blocksDevice, blocksMemorySize);     
        MANAGE_ERRORS("ERROR:Failed to cuda malloc blocksDevice in HOG::initialize\n");
        cudaMemGetInfo(&f, &t);
        std::cout <<"after cudaMalloc " <<"free:" << f/1024*1024 << " bytes total:" <<t/1024*1024<<" bytes"<<"\n";
        return 0;
    }
    int HOG::freeImage()
    {
         return eraseImage();
    }
    int HOG::finalize()
    {
        cudaFree(gradientsDevice);
        gradientsDevice = NULL;
        MANAGE_ERRORS("ERROR:Failed to cuda free gradientsDevice in HOG::finalize()\n");
        cudaFree(blocksDevice);
        blocksDevice = NULL;
        MANAGE_ERRORS("ERROR:Failed to cuda free blocksDevice in HOG::finalize()\n");
        if( finalizeSVM() || finalizeComputeBlocks() )
        {
            return -1;
        }
        return 0;
    }
    int HOG::transferImage(const unsigned char *imageHost, int width, int height)
    {
        if(prepareImage(imageHost,width,height))
        {
            std::cout<<"ERROR: prepareImage in HOG::transferImage"<<"\n";
            return -1;
        }
        return 0;
    }
    int HOG::processImageOneScale(int width, int height, float scale,int padX, int padY, int* cntBlocks, int* cntSVM
                                  ,multiDetectionList& detectionList)
        {
            // [Zaid-11] cambio el dia 25/10/2014
//            printf("[-----HOG---Inicio de hog_process_image-----------------]\n");
            // Fin [Zaid-11]
            Timer tGradients;
            Timer tBlocks;
            Timer tSVM;
            Timer tEnd;
            int result;
            startTimer(&tGradients);
            if(padX == -1 || padY == -1)
            {
                padX = PADDING_X;
                padY = PADDING_Y;
            }
            int paddedWidth = padX * 2 + width;
            int paddedHeight= padY * 2 + height;
            int minX = 0;
            int minY = 0;
            int maxX = width;
            int maxY = height;
            int gradWidth = (maxX-minX) +2*padX -1;
            int gradHeight = (maxY-minY)+2*padY -1;
            //cudaMemset(gradientsDevice, 0, sizeof(float2) * MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION);            
	    cudaMemset(gradientsDevice, 0, sizeof(float2) * gradHeight * gradWidth);
            MANAGE_ERRORS("ERROR:Failed to cuda memset gradientsDevice in HOG::processImageOneScale\n");
            result = calculateGradientsHost(paddedWidth, paddedHeight, minX, minY, maxX, maxY, padX, padY, gradientsDevice);
            cudaDeviceSynchronize();
	    if(result)
            {
                std::cout<<"ERROR:Failed to calculateGradientsHost in HOG::processImageOneScale:"<< result << "\n";
                return -1;
            }            
            stopTimer(&tGradients);
            startTimer(&tBlocks);
            dim3 blockGrid;
            blockGrid.x = (int)floor(gradWidth / 8.f);
            blockGrid.y = (int)floor(gradHeight / 8.f);
            result = computeBlocksHost(gradWidth, gradHeight, gradientsDevice, blocksDevice);
            if(result)
            {
                std::cout<<"ERROR:Failed to computeBlocksHost in HOG::processImageOneScale:"<< result << "\n";
                return -2;
            }
            //printf("valores de sigmaX,sigmaY,sigmaS----:%f----%f----%f\n",sigmaX,sigmaY,sigmaS);
            *cntBlocks += blockGrid.x * blockGrid.y;
            stopTimer(&tBlocks);
            startTimer(&tSVM);
            int count;
            result = evaluateModelsSVM(blocksDevice, blockGrid.x, blockGrid.y, padX, padY, minX, minY, scale, &count, detectionList);
            if(result)
            {
                std::cout<<"ERROR:Failed to evaluateModels in HOG::processImageOneScale:"<< result << "\n";
                return -3;
            }
            *cntSVM += count;
            stopTimer(&tSVM);
            startTimer(&tEnd);
            stopTimer(&tEnd);
            totalGradients += getTimer(&tGradients);
            totalBlocks += getTimer(&tBlocks);
            totalSvm += getTimer(&tSVM);
            totalEnd += getTimer(&tEnd);
            // [Zaid-11] cambio el dia 25/10/2014
//            printf("[-----HOG---Fin de hog_process_image--------------------]\n");
            // Fin [Zaid-11]
            return 0;
        }


    int HOG::processImageMultiScale(multiDetectionList& detectionList,int width, int height, int* blocks, int* cntSVM
                                    ,double* timings)
    {
        // [Zaid-11] cambio el dia 25/10/2014
//        printf("[-----HOG---Inicio de hog_process_image_multiscale-------]\n");
        // Fin [Zaid-11]
        Timer globalTimer;
        startTimer(&globalTimer);
        totalGradients=  0.;
        totalBlocks = 0.;
        totalSvm = 0.;
        totalEnd= 0.;
        multiDetectionList finalDetectionList;
        float startScale = START_SCALE;
        int minWindowWidth = globalParameters.computeMinWindowWidth();
        int minWindowHeight = globalParameters.computeMaxWindowHeight();
        float endScale = min((width + 2*PADDING_X ) / (float)minWindowWidth,
                                (height+ 2*PADDING_Y ) / (float)minWindowHeight);
        std::vector<float> scales;        
        float scale = startScale;
        size_t i=0;
        while( scale < endScale )
        {
            scales.push_back(scale);
            scale *= (float)SCALE_STEP;
            i++;
        }
        int count=0;
        for(; count < scales.size(); count++)
        {
            const float scale = scales[count];
            int currentWidth = width / scale;
            int currentHeight = height / scale;
//            std::cout <<"currentWidth "<<currentWidth<<"MAX "<<MAX_IMAGE_DIMENSION<<std::endl;
            assert(currentWidth < MAX_IMAGE_DIMENSION );
            assert(currentHeight < MAX_IMAGE_DIMENSION );            
            int validModel = 0;
            int i;
                for(i = 0; i < globalParameters.configModels.size(); i++ )
                {
                    if(scale >= globalParameters.configModels[i].minScale
                            || scale <= globalParameters.configModels[i].maxScale )
                    {
                        validModel = 1;
                    }
                }
                processImageOneScale(currentWidth, currentHeight, scale,PADDING_X, PADDING_Y, blocks,
                                     cntSVM, finalDetectionList);
        }
        Timer nmsTimer;
        NMS nms;
        startTimer(&nmsTimer);
        detectionList.resize(finalDetectionList.size());

        if(!NON_MAXIMUM_SUPPRESSION )
        {
            std::copy(finalDetectionList.begin(), finalDetectionList.end(), detectionList.begin());
        }
        else
        {
            for(int i=0; i < finalDetectionList.size(); i++)
            {
               nms.processDetectionList((finalDetectionList[i]) , (detectionList[i]),  globalParameters.configModels[i]);
            }
        }     
        stopTimer(&nmsTimer);
        stopTimer(&globalTimer);
        double totalNms = getTimer(&nmsTimer);
        timings[0] += totalGradients;
        timings[1] += totalBlocks;
        timings[2] += totalSvm;
        timings[3] += totalNms;
        timings[4] += getTimer(&globalTimer);
        finalDetectionList.clear();
        // [Zaid-11] cambio el dia 25/10/2014
//        printf("[-----HOG---Fin de hog_process_image_multiscale-------]\n");
        // Fin [Zaid-11]
        return 0;
    }  
}

