#ifndef __NMS_H__
#define __NMS_H__
#include "HOGPlanner.h"
#include "DumpDetections.h"
#include "GlobalParameters.h"
namespace UC3M_HOG_GPU
{
    typedef struct
    {
        float x;
        float y;
        float scale;
        float score;
    } nmsDetection;
    class NMS
    {
        public:
            void processDetectionList(detectionList& detectionsList, detectionList& detectionsNMS, LocalParameters &params);
    };
}
#endif
