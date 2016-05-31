#ifndef _LOCAL_PARAMETERS_H_
#define _LOCAL_PARAMETERS_H_
#include <vector>
#include <string>
namespace UC3M_HOG_GPU
{
    class LocalParameters
    {
        public:
            LocalParameters(int descriptorWidth=7, int descriptorHeight= 15, int windowWidth=64, int windowHeight=128)
                        : id("no_name"), filename(""), DESCRIPTOR_WIDTH(descriptorWidth), DESCRIPTOR_HEIGHT(descriptorHeight),
                        WINDOW_WIDTH(windowWidth),WINDOW_HEIGHT(windowHeight),minScale(0.f), maxScale(100000.f)
            {};
        public:
            std::string id;
            std::string filename;
            int DESCRIPTOR_WIDTH;
            int DESCRIPTOR_HEIGHT;
            int WINDOW_WIDTH;
            int WINDOW_HEIGHT;
            float minScale;
            float maxScale;
            int getDimension();

    };
}
#endif
