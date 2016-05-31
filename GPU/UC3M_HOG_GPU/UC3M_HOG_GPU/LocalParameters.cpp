#include <cassert>
#include <climits>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include "Global.h"
#include "LocalParameters.h"
namespace UC3M_HOG_GPU
{
    int LocalParameters::getDimension()
    {
        return DESCRIPTOR_HEIGHT * DESCRIPTOR_WIDTH * NBINS * NUM_BLOCK_CELLS_X * NUM_BLOCK_CELLS_Y;
    }
}
