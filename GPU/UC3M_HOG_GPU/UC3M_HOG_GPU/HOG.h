#include "DumpDetections.h"
#include "HOGPlanner.h"
namespace UC3M_HOG_GPU
{
    class HOG
    {
    public:
        int initialize();
        int finalize();
        int transferImage(const unsigned char* imageHost, int width, int height);
        int freeImage();
        int processImageOneScale(int width, int height, float scale, int padX, int padY, int* cntBlocks, int* cntSVM, multiDetectionList& detectionList);
        int processImageMultiScale(multiDetectionList& detects, int width, int height, int* blocks, int* cntSVM,
                                    double* timings);
        int getDescriptor(int width, int height, int bPad,
                            int featureX, int featureY, float scale,
                            LocalParameters& params,
                            float* h_pDescriptor);
};
}

