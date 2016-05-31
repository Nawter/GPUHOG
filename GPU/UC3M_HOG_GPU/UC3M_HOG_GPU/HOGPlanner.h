#ifndef ___HOG_PLANNER_H__
#define ___HOG_PLANNER_H__
#include <string>
#include <vector>
#include <climits>
#include "Global.h"
#include "DumpDetections.h"
namespace UC3M_HOG_GPU
{
    class LocalParameters;
    class GlobalParameters;
    using namespace std;
    class HOGPlanner
    {
        public:
            HOGPlanner();           
            ~HOGPlanner();
            int transferImage(uchar* image, ushort width, ushort height);
            int readFromConfigFile(std::string& file);           
            int freeImage();
            int readFromModelFile();
            int detectInImages(const vector<string>& images, const vector<string>& output,
                            int* blocks=NULL, int* cntSVM=NULL, double* timings=NULL);
    private:
            bool imageReady;
            bool isModel;
            ushort imageWidth, imageHeight;
            //LocalParameters* selectedModels;


    };
}
#endif
