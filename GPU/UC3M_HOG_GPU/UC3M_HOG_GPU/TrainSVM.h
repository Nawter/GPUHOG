#ifndef __TRAIN_SVM_H__
#define __TRAIN_SVM_H__
#include "HOGPlanner.h"
#include "DumpDetections.h"

namespace UC3M_HOG_GPU
{
class Model {
public:
    Model() :weights(0), bias(0), weightsHost(NULL), weightsDevice(NULL) {}
    long weights;
    double bias;
    float* weightsHost;
    float* weightsDevice;
};

    int initializeSVM();
    int finalizeSVM();
    int readFromModelFileSVM(const char* model);
    int evaluateModelsSVM(float* blocksDevice, int blocksX, int blocksY,int padX, int padY, int minX, int minY,
                              float scale,int* svm, multiDetectionList& detectionList);
    int readFromBinaryFileSVM(const char* svmModel, Model& model);


}
#endif
