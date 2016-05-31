#include <ctype.h>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
//#include "opencv2/gpu/gpu.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat img,img_aux;    
    cuda::GpuMat gpu_img;
	double scale = 1.05;
    int nlevels = 64;
    int gr_threshold = 1;
    int hit_threshold = 0;
    int win_width = 64;
    int win_stride_width = 8;
    int win_stride_height = 8;   
    FILE* f = 0;
    char _filename[1024];
    if( argc == 1 )
    {
        printf("Usage: OPENCV_HOG_GPU (<image_filename> | <image_list>.txt)\n");
        return 0;
    }
    img = imread(argv[1]);

    if( img.data )
    {
        strcpy(_filename, argv[1]);
    }
    else
    {
        f = fopen(argv[1], "rt");
        if(!f)
        {
            fprintf( stderr, "ERROR: the specified file could not be loaded\n");
            return -1;
        }
    }
     // Create HOG descriptors and detectors here
    Size win_size(64, 64 * 2); //(64, 128) or (48, 96)
    Size win_stride(8, 8);    
    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size);
    // Create HOG descriptors and detectors here
    Mat detector = gpu_hog->getDefaultPeopleDetector();
    gpu_hog->setSVMDetector(detector);
    for(;;)
    {
        char* filename = _filename;
        if(f)
        {
            if(!fgets(filename, (int)sizeof(_filename)-2, f))
                break;
            if(filename[0] == '#')
                continue;
            int l = (int)strlen(filename);
            while(l > 0 && isspace(filename[l-1]))
                --l;
            filename[l] = '\0';
            img = imread(filename);
        }
        if(!img.data)
            continue;
        fflush(stdout);
        vector<Rect> found, found_filtered;
     	cvtColor(img, img_aux, COLOR_BGR2GRAY);
       	gpu_img.upload(img_aux);
        gpu_hog->setNumLevels(nlevels);
        gpu_hog->setHitThreshold(hit_threshold);
        gpu_hog->setWinStride(win_stride);
        gpu_hog->setScaleFactor(scale);
        gpu_hog->setGroupThreshold(gr_threshold);
        gpu_hog->detectMultiScale(gpu_img, found);
        size_t i, j;
   //      for( i = 0; i < found.size(); i++ )
   //      {
   //  	    Rect r = found[i];
   //  	    rectangle(img, r.tl(), r.br(), cv::Scalar(110,155,255), 1);
   //  	    for( j = 0; j < found.size(); j++ )
   //  	    {    
   //      	 	if( j != i && (r & found[j]) == r)
   //      		{
   //      	        break;
   //      		}
	  //       }
   //  	    if( j == found.size() )
   //  	    {	  
   //  		  found_filtered.push_back(r);
   //  	    }
	  //  }
	  // for( i = 0; i < found_filtered.size(); i++ )
   //      {
   //          Rect r = found_filtered[i];
   //          // the HOG detector returns slightly larger rectangles than the real objects.
   //          // so we slightly shrink the rectangles to get a nicer output.
   //          //r.x += cvRound(r.width*0.1);
   //          //r.width = cvRound(r.width*0.8);
   //          //r.y += cvRound(r.height*0.07);
   //          //r.height = cvRound(r.height*0.8);	 
	  //   rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
   //      }
        
        waitKey(0);
            break;        
    }
    //getHumanReadableTime(start);
    if(f)
    fclose(f);
    return 0;
}
