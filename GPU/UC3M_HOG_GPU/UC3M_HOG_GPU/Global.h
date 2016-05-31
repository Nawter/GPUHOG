#ifndef __GLOBAL_h__
#define __GLOBAL_h__
#ifndef uchar
#define uchar unsigned char
#endif
#ifndef ushort
#define ushort unsigned short
#endif
// BEWARE: the initial mallocs are done according to this maximum requirement
const int MAX_IMAGE_DIMENSION = 3000;
const int SVM_MAX_MODELS = 10;
// NMS mean-shift parameters
const int 	SIGMA_FACTOR 	= 4;
const float	SIGMA_SCALE		= 1.6f;
#define NUM_BLOCK_CELLS_X	2
#define NUM_BLOCK_CELLS_Y	2
#define BLOCK_WIDTH 	(8*NUM_BLOCK_CELLS_X)
#define BLOCK_HEIGHT	(8*NUM_BLOCK_CELLS_Y)
#define CELL_SIZE		8		// we assume 8x8 cells!
#define PADDING_X		16		// padding in pixels to add to each side
#define PADDING_Y		16		// padding in pixels to add to top&bottom
#define NEGATIVE_SAMPLES_PER_IMAGE	10
const float	START_SCALE = 1.0f;
const float SCALE_STEP  = 1.05f;
const float SIGMA	=		(0.5 * BLOCK_WIDTH);		// gaussian window size for block histograms
const int NBINS = 9;
const float TRAINING_SCALE_STEP = 1.2f; // the original procedure uses 1.2 steps
const int MAXIMUM_HARD_EXAMPLES = 50000;
// 0 -- use weighted sum score for mode
// 1 -- use maximum score for mode
#define NMS_MAXIMUM_SCORE	1
#define NON_MAXIMUM_SUPPRESSION  1 // 1 NMS enabled (default) -- 0 NMS disabled
#define NMS_MAX_ITERATIONS	100
#define NMS_MODE_EPSILON	((float)1E-5f)
#define NMS_FINAL_DIST		1.f
// FLAGS
#define ENABLE_GAMMA_COMPRESSION	// enable sqrt gamma compression
#define PRINT_PROFILING_TIMINGS		0 // show profiling timings for each frame
#define PRINT_VERBOSE_INFO			0 // show detection information
#define PRINT_DEBUG_INFO			0 // show verbose debug information at each scale level
// DEBUG FLAGS
#define DEBUG_PRINT_PROGRESS					0
#define DEBUG_PRINT_SCALE_CONSTRAINTS 			0
#define VERBOSE_CUDA_FAILS
#ifdef VERBOSE_CUDA_FAILS
#define MANAGE_ERRORS(S) \
{ \
    cudaError_t error = cudaGetLastError(); \
    if(error) \
    { \
        printf("%s:%s:"S":%s\n", __FILE__, __FUNCTION__ , cudaGetErrorString(error));\
        return -2;\
    } \
}
#else
#define MANAGE_ERRORS(S)
#endif
#endif
