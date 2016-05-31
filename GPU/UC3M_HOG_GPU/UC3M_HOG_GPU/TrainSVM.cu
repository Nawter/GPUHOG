#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <iostream>


#include "HOGPlanner.h"
#include "GlobalParameters.h"
#include "HOGPlanner.h"
#include "DumpDetections.h"
#include "TrainSVM.h"


texture<float, 1, cudaReadModeElementType> textureWeights;

float* resultsDevice = NULL;
float* resultsHost = NULL;

namespace UC3M_HOG_GPU
{
    std::vector<Model> svmModels;
    int initializeSVM()
        {
        cudaMalloc(&resultsDevice,
                            sizeof(float) *
                            (MAX_IMAGE_DIMENSION/8) 
                            *(MAX_IMAGE_DIMENSION/8));
        MANAGE_ERRORS("ERROR:Failed to cuda malloc deviceResults in SVM::initialize\n");
        cudaMemset(resultsDevice, 0, sizeof(float) *
                            (MAX_IMAGE_DIMENSION/8)*
                           (MAX_IMAGE_DIMENSION/8));
        MANAGE_ERRORS("ERROR:Failed to cuda memset deviceResults in SVM::initialize\n");
        resultsHost = (float*)malloc(sizeof(float)*
                                    (MAX_IMAGE_DIMENSION/8)* 
                                  (MAX_IMAGE_DIMENSION/8));
        return 0;
        }
    int finalizeSVM()
    {
        for(size_t x=0; x < svmModels.size(); x++)
        {
            //if(!svmModels[ii].weightsHost)
              //  continue;
            if(svmModels[x].weightsHost)
            {
            free(svmModels[x].weightsHost);
            cudaFree(svmModels[x].weightsDevice);
            }
        }
        svmModels.clear();
        free(resultsHost);
        return 0;
    }
    int readFromBinaryFileSVM(const char* svmModel, Model& model)
    {
      FILE *fp;
      int verbosity = 0;
      if(verbosity>=2)
      {
        std::cout << "Reading model ...";
        fflush(stdout);
      }
      if ((fp = fopen (svmModel, "rb")) == NULL)
      {
          std::cout << "ERROR:failed in fopen in readFromBinaryFile:" << svmModel << "\n";
          return -1;
      }
      char versionBuffer[10];
      if (!fread (&versionBuffer,sizeof(char),10,fp))
      {
          std::cout << "ERROR:failed in read version in readFromBinaryFile:" << "\n";
          return -1;
      }
      int version = 0;
      if (!fread (&version,sizeof(int),1,fp))
      {
          std::cout << "ERROR:failed in read version number in readFromBinaryFile:" << "\n";
          return -1;
      }
      if (version < 200)
      {
          std::cout << "Model file compiled for light version:" << "\n";
          return -1;
      }
      fseek(fp, sizeof(int64_t) * 2  + sizeof(double) * 3, SEEK_CUR);
      int64_t l;
      fread(&l,sizeof(int64_t),1,fp);
      fseek(fp, l*sizeof(char), SEEK_CUR);
      fread(&(model.weights),sizeof(int64_t),1,fp);
      fseek(fp, sizeof(int64_t), SEEK_CUR);
      if(version >= 201)
      {
        fseek(fp, 2*sizeof(double), SEEK_CUR);
      }

      fseek(fp, sizeof(int64_t), SEEK_CUR);
      fread(&(model.bias), sizeof(double),1,fp);

      model.weightsHost =(float*)malloc(sizeof(float)*(model.weights+1));
      if(model.weightsHost==NULL)
      {
          std::cout << "ERROR:weightsHost is NULL in readFromBinaryFile:" << "\n";
          return -1;
      }
      double* tmpWeights  = (double*)malloc(sizeof(double)*(model.weights+1));
      assert(tmpWeights  != NULL);
      if(tmpWeights==NULL)
      {
          std::cout << "ERROR:tmpweights is NULL in readFromBinaryFile:" << "\n";
          return -1;
      }
      fread(tmpWeights, sizeof(double),model.weights+1,fp);
      double* tmpPtr = tmpWeights;
      float* mPtr = model.weightsHost;
      for(int j=0; j < model.weights+1; j++, tmpPtr++, mPtr++)
      {
          *(mPtr) = (float)(*tmpPtr);
      }
      free(tmpWeights);
      fclose(fp);
      return 0;
    }
    int readFromModelFileSVM(const char *model)
    {
        Model svmModel;
        if(!model)
        {
            svmModel.weightsHost = NULL;
            svmModel.weights= 0;
            svmModel.bias = 0;
        }
        else
        {
            if(readFromBinaryFileSVM(model, svmModel) )
            {
                std::cout<<"ERROR: readFromBinaryFile failed\n";
                return -1;
            }

        }
        cudaMalloc((void**)&(svmModel.weightsDevice), sizeof(float) * svmModel.weights);
        MANAGE_ERRORS("ERROR:Failed to cuda malloc svmModel.weightsDevice in SVM::readFromModelFile\n");
        cudaMemset(svmModel.weightsDevice, 0, sizeof(float) * svmModel.weights);
        MANAGE_ERRORS("ERROR:Failed to cuda memset svmModel.weightsDevice in SVM::readFromModelFile\n");
        cudaMemcpy(svmModel.weightsDevice, svmModel.weightsHost, sizeof(float) * svmModel.weights, cudaMemcpyHostToDevice);
        MANAGE_ERRORS("ERROR:Failed to cuda memcpy svmModel.weightsDevice in SVM::readFromModelFile\n");
        svmModels.push_back(svmModel);
        return 0;
    }
    extern "C++"__global__ void evaluateOneModelDevice(float* blocksDevice, float bias, int blocksX, float* resultsDevice)

    {
        const int index = threadIdx.x;
        volatile __shared__ float resultsShared[1024];
        resultsShared[index] = 0.f;
        const int dX = blockIdx.x;
        const int dY = blockIdx.y;
        const int floatsPerDescriptorRow = 7 * 4 * 9;
        if( index < 252 )
        {
            for(int row=0; row < 15; row++)
            {
                const int weightsIndex = row * floatsPerDescriptorRow;
                float* block = blocksDevice + 36 * ((dY+row) * blocksX + dX);
                resultsShared[index] += tex1Dfetch(textureWeights, weightsIndex + index) * block[index];
            }
        }
        __syncthreads();
        if( index < 128 )
        {
            resultsShared[index] += resultsShared[index + 128];
        }
        __syncthreads();
        if( index < 64 )
        {
            resultsShared[index] += resultsShared[index + 64];
        }
        __syncthreads();
        if( index < 32 )
        {
            resultsShared[index] += resultsShared[index + 32];
            resultsShared[index] += resultsShared[index + 16];
            resultsShared[index] += resultsShared[index + 8];
            resultsShared[index] += resultsShared[index + 4];
            resultsShared[index] += resultsShared[index + 2];
            resultsShared[index] += resultsShared[index + 1];
        }
        if( index == 0 )
        {
            resultsDevice[blockIdx.y * gridDim.x + blockIdx.x] = resultsShared[0] - bias;
        }
    }
    extern "C++"__global__ void evaluateOneModelDevice(float* blocksDevice, float bias,int blocksX, int descriptorWidth, int descriptorHeight
                                   ,float* resultsDevice)

    {
        const int index = threadIdx.x;
        __shared__ float resultsShared[128];
        resultsShared[index] = 0.f;
        const int dX = blockIdx.x;
        const int dY = blockIdx.y;
        const int floatsPerDescriptorRow = descriptorWidth * NUM_BLOCK_CELLS_X * NUM_BLOCK_CELLS_Y * NBINS;
        const int floatsPerBlock = NUM_BLOCK_CELLS_X * NUM_BLOCK_CELLS_Y * NBINS;
        const int offset = 9 * index;
        if( offset < floatsPerDescriptorRow )
        {
            for(int row=0; row < descriptorHeight; row++)
            {
                const int weightsIndex = row * floatsPerDescriptorRow;
                float* block = blocksDevice + floatsPerBlock * ((dY+row) * blocksX + dX);
                for(int i=0; i < 9; i++)
                {
                    resultsShared[index] += tex1Dfetch(textureWeights, weightsIndex + offset + i) * block[offset+i];
                }
            }
        }
        __syncthreads();
        if( index < 64 )
        {
            resultsShared[index] += resultsShared[index + 64];
        }
        __syncthreads();
        if( index < 32 )
        {
            resultsShared[index] += resultsShared[index + 32];
            resultsShared[index] += resultsShared[index + 16];
            resultsShared[index] += resultsShared[index + 8];
            resultsShared[index] += resultsShared[index + 4];
            resultsShared[index] += resultsShared[index + 2];
            resultsShared[index] += resultsShared[index + 1];
        }
        if( index == 0 )
        {
            resultsDevice[blockIdx.y * gridDim.x + blockIdx.x] = resultsShared[0] - bias;
        }
    }
    extern "C++"__host__ int evaluateOneModelHost(int modelIdx, float *blocksDevice, int blocksX, int blocksY, int *cntSVM, float* resultsHost)
    {
        const int descriptorWidth = globalParameters.configModels[modelIdx].DESCRIPTOR_WIDTH;
        const int descriptorHeight = globalParameters.configModels[modelIdx].DESCRIPTOR_HEIGHT;
        const int descriptorsX = blocksX - descriptorWidth +1;
        const int descriptorsY = blocksY - descriptorHeight +1;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  //      cudaMalloc((void**)&resultsDevice,sizeof(float) * (MAX_IMAGE_DIMENSION/8) *(MAX_IMAGE_DIMENSION/8));
//	MANAGE_ERRORS("ERROR:Failed to cuda malloc deviceResults in SVM::initialize\n");
        cudaBindTexture(0, textureWeights,
                        svmModels[modelIdx].weightsDevice,
                        channelDesc,sizeof(float) * svmModels[modelIdx].weights);
        MANAGE_ERRORS("ERROR:Failed to bind textureWeights in SVM::evaluateOneModelHost\n");
        memset(resultsHost, 0, sizeof(float) * descriptorsX * descriptorsY);
        MANAGE_ERRORS("ERROR:Failed to cuda memset deviceResults in SVM::initialize\n");
        if(descriptorWidth == 7  &&	descriptorHeight == 15 )
        {
            dim3 grid(descriptorsX, descriptorsY, 1);
            dim3 threads(256, 1, 1);
            //dim3 threads(32, 1, 1);
            evaluateOneModelDevice<<< grid, threads>>>(blocksDevice, svmModels[modelIdx].bias, blocksX, resultsDevice);
        }
        else
        {
            const int nThreads = 128;
            //const int nThreads = 16;
            if( descriptorWidth * 4 > nThreads )
            {
                std::cout<<"WARNING:evaluateOneModelHost will not work\n";
                return -1;
            }
            dim3 grid(descriptorsX, descriptorsY, 1);
            dim3 threads(nThreads, 1, 1);
            evaluateOneModelDevice<<<grid, threads>>>(blocksDevice, svmModels[modelIdx].bias, blocksX,descriptorWidth, descriptorHeight, resultsDevice);
        }
        //*cntSVM = descriptorsX * descriptorsY;
	const int z=descriptorsX * descriptorsY;
	//std::cout<<"ZZZZZZZZZZZZZ:::"<<z<<"\n";
	cudaMemcpy(resultsHost, resultsDevice, sizeof(float) * z, cudaMemcpyDeviceToHost);        
        MANAGE_ERRORS("ERROR:Failed to cuda memcpy to resutlsHost in SVM::evaluateOneModelHost\n");
        cudaUnbindTexture(textureWeights);
        MANAGE_ERRORS("ERROR:Failed to unbind textureWeights in SVM::evaluateOneModelHost\n");
        return 0;
    }
    int evaluateModelsSVM(float* blocksDevice, int blocksX, int blocksY,int padX, int padY, int minX, int minY,
                          float scale,int* svm, multiDetectionList& detectionList)
    {
        // [Zaid-11] cambio el dia 25/10/2014
//        printf("[-----TrainSVM---Inicio de svm_evaluate-----]\n");
        // Fin [Zaid-11]
        assert(svmModels.size() > 0);
        detectionList.resize(svmModels.size());
        for(size_t i=0; i < svmModels.size(); i++)
        {
            const int width = globalParameters.configModels[i].DESCRIPTOR_WIDTH;
            const int height =globalParameters.configModels[i].DESCRIPTOR_HEIGHT;
            if( blocksX <= width || blocksY <= height )
            {
                //std::cout << "evaluateModels: Skipping model" << globalParameters.configModels[i].id.c_str() << "at scale" << scale << "\n";
                continue;
            }
           if(  scale > globalParameters.configModels[i].maxScale
                || scale < globalParameters.configModels[i].minScale )
            {
               //std::cout << "evaluateModels: Skipping model" << i << "at scale" << scale << "\n";
               continue;               
            }
            else
            {
               //std::cout << "evaluateModels: Skipping model" << globalParameters.configModels[i].id.c_str() << "at scale" << scale << "\n";
            }
            if(evaluateOneModelHost(i, blocksDevice, blocksX, blocksY, svm, resultsHost))
            {
                std::cout<<"ERROR:Failed to evaluateOneModelHost in SVM::evaluteModels:\n";
                return -1;
            }
            const int descriptorsX = blocksX - width +1;
            const int descriptorsY = blocksY - height +1;
            for(int y=0; y < descriptorsY; y++)
            {
                for(int x=0; x < descriptorsX; x++)
                {
                    float score = resultsHost[y*descriptorsX + x];
                    if( score > 0.f )
                    {
                        // [Zaid-11] cambio el dia 25/10/2014
//                        printf("---TrainSVM---evaluateModelsSVM---El valor de detectionList[i].size():::%d++++\n",detectionList[i].size());
                        // Fin [Zaid-11]
                        int posX = (x * CELL_SIZE - padX + minX) * scale ;
                        int posY = (y * CELL_SIZE - padY + minY) * scale ;                       
                        detectionList[i].push_back(Detection(posX, posY, scale, score));
                    }
                }
            }
        }
        // [Zaid-11] cambio el dia 25/10/2014
//        printf("[-----TrainSVM---Fin de svm_evaluate-----]\n");
        // Fin [Zaid-11]
        return 0;
    }
}
