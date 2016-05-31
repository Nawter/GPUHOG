#ifndef _GLOBAL_PARAMETERS_H_
#define _GLOBAL_PARAMETERS_H_
#include <vector>
#include <string>
#include "LocalParameters.h"
namespace UC3M_HOG_GPU
{
class GlobalParameters
{
      public:
        int getMinDescriptorHeight();
        int getMinDescriptorWidth();
        int computeMinWindowHeight();
        int computeMinWindowWidth();
        int computeMaxWindowHeight();
        int computeMaxWindowWidth();
        int loadFromFile(std::string& file);
        std::string location;
        std::vector<LocalParameters> configModels;

};
extern GlobalParameters globalParameters;
}
#endif
