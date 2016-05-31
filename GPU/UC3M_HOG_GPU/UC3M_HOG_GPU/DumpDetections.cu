#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "DumpDetections.h"
#include "HOGPlanner.h"
#include "GlobalParameters.h"
namespace UC3M_HOG_GPU
{
    Detection::Detection()
    {
        x=INT_MAX;
        y=INT_MAX;
        scale=0.f;
        score=0.f;
    }
    Detection::Detection(int nx, int ny, float nscale, float nscore)
    {
        x = nx;
        y = ny;
        scale = nscale;
        score = nscore;
    }
    void dumpMultiDetectionList(multiDetectionList& detectionsList, const std::string& fn)
    {
        assert(globalParameters.configModels.size() > 0);
        FILE *file = fopen(fn.c_str(), "w");
        for(size_t j=0; j < detectionsList.size(); j++)
        {
            fprintf(file, "\n#--------------------------------\n");
            fprintf(file, "#  model: %d\n#\n", j+1);
            const int detectionWidth = globalParameters.configModels[j].WINDOW_WIDTH;
            const int detectionHeight = globalParameters.configModels[j].WINDOW_HEIGHT;
            for(size_t i=0; i < detectionsList[j].size(); i++)
            {
                Detection d = detectionsList[j][i];                
                int x = d.x + detectionWidth * d.scale;
                int y = d.y + detectionHeight * d.scale;              
                fprintf(file, "%d\t%d\t%d\t%d\t%f\n", d.x, d.y, x, y, d.score);
            }
        }
        fclose(file);
    }
}
