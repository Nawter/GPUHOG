#ifndef __DUMP_DETECTIONS_H__
#define __DUMP_DETECTIONS_H__
#include <vector>
#include <string>
#include "HOGPlanner.h"
namespace UC3M_HOG_GPU
{
    class Detection
    {
        public:
            Detection();
            Detection(int nx, int ny, float nscale, float nscore);
        public:
            int x;
            int y;
            float scale;
            float score;
    };
    typedef std::vector<Detection> detectionList;
    typedef std::vector<detectionList> multiDetectionList;     
    void dumpMultiDetectionList(multiDetectionList& detectionsList, const std::string& fn);
}
#endif
