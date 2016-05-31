#include <iostream>
#include <fstream>
#include <QImage>
#include <cmath>
#include <deque>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include "HOGPlanner.h"
#include "HOG.h"
#include "TrainSVM.h"
#include "Global.h"
#include "GlobalParameters.h"
#include "LocalParameters.h"
#include "DumpDetections.h"
#include "ext/Vector.h"
#include "ext/Matrix.h"
namespace UC3M_HOG_GPU
{
    using namespace std;
    GlobalParameters globalParameters;
    HOG hog;
    // Cambiar Detection por Dump y crear una nueva clase Detection
    //Detection dump;
    HOGPlanner::HOGPlanner()
        : imageReady(false),isModel(true),
          imageWidth(-1), imageHeight(-1)
          //selectedModels(NULL)
    {
        globalParameters.configModels.clear();
        if(hog.initialize())
        {
           std::cout << "ERROR:hog.initialize() failed.\n";
        }
        if(initializeSVM())
        {
            std::cout << "ERROR:svm.initialize() failed.\n";
        }
    }
    HOGPlanner::~HOGPlanner()
    {
        if(imageReady)
        {
            hog.freeImage();
        }
        if(hog.finalize())
        {
            std::cout << "ERROR:hog.finalize() failed.\n";
        }
    }
    int HOGPlanner::readFromConfigFile(std::string& file)
    {
        globalParameters.configModels.clear();
        return globalParameters.loadFromFile(file);
    }
    int HOGPlanner::readFromModelFile()
    {
        if(globalParameters.location.size() == 0)
        {
            std::cout << "ERROR:readFromModelFile() failed.\n";
            return -1;
        }
//        std::cout << "location.size:" << globalParameters.location.size() <<"\n";
//        printf("# models: %d\n", globalParameters.configModels.size());
        for(size_t x=0; x < globalParameters.configModels.size(); x++ )
        {
//            std::cout << "id.size:" << globalParameters.configModels[x].id.size() <<"\n";
            if(globalParameters.configModels[x].id.size() == 0)
            {
                std::cout << "ERROR:readFromModelFile() failed.\n";
                return -1;
            }
            string model;
            if(globalParameters.configModels[x].filename.length() > 0 )
            {
//                std::cout << "location:" << globalParameters.location<<"\n";
//                std::cout << "id:" << globalParameters.configModels[x].id<<"\n";
//                std::cout << "filename:" << globalParameters.configModels[x].filename<<"\n";
                model = globalParameters.location +string("/")+ globalParameters.configModels[x].filename;
            }
            else
            {
                model = globalParameters.location +string("/")+ globalParameters.configModels[x].id;
            }
//            printf("loading model: %s\n", model.c_str());
           if(readFromModelFileSVM(model.c_str()))
            {
                return -1;
            }
        }
        return 0;
    }
//    int HOGPlanner::freeImage()
//    {
//        return 0;
//    }
    int HOGPlanner::transferImage(uchar* image, ushort width, ushort height)
    {
        if(imageReady)
        {
             hog.freeImage();
        }
        imageWidth = width;
        imageHeight = height;
        if(hog.transferImage(image, imageWidth, imageHeight))
        {
            std::cout<<"ERROR: hog.transferImage failed"<<"\n";
            return -1;
        }
        imageReady = true;
        return 0;
    }
    int HOGPlanner::detectInImages( const std::vector<string>& images,
                                    const std::vector<string>& output,
                                    int* blocks,
                                    int* cntSVM,
                                    double* timings)
    {
        // [Zaid-11] cambio el dia 25/10/2014
//        printf("[----UC3M_HOG_GPU---Inicio de test_images-------------]\n");
        // Fin [Zaid-11]
        int totalBlocks = 0;
        int totalSVM = 0;
        if(blocks == NULL || cntSVM == NULL)
        {
            blocks = &totalBlocks;
            cntSVM = &totalSVM;
        }
        for(size_t i=0; i < images.size(); i++)
        {
            QImage image;
//            printf("---->%s\n",images[i].c_str());
            image.load(images[i].c_str());
            if(image.isNull())
            {
                std::cout << "ERROR:failed to load image in HOGPlanner" << images[i].c_str() <<"\n";
                return -1;
            }
            // Porque se hace esta conversion hacia este formato hacer pruebas con y sin ese formato.
            QImage convertedImage = image.convertToFormat(QImage::Format_ARGB32);
            if(transferImage(convertedImage.bits(), convertedImage.width(),convertedImage.height()))
            {
                std::cout << "ERROR:failed in transferImage in HOGPlanner" << images[i].c_str() <<"\n";
                return -1;
            }
            multiDetectionList detectionList;
            if(hog.processImageMultiScale(detectionList,imageWidth, imageHeight,blocks,cntSVM,timings))
            {
                std::cout << "ERROR:failed in hog.processImageMultiScale in HOGPlanner" << images[i].c_str() <<"\n";
                return -1;
            }         
            //dumpMultiDetectionList(detectionList, output[i]);
            detectionList.clear();
            hog.freeImage();
            imageReady = false;            
        }
        // [Zaid-11] cambio el dia 25/10/2014
//        printf("[----UC3M_HOG_GPU---Fin de test_images-------------]\n");
        // Fin [Zaid-11]
        return 0;
    }
}
