#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <HOGPlanner.h>
#include <sys/time.h>
using namespace std;
using namespace UC3M_HOG_GPU;
int getMilliSecond()
{
    static int started = 0;
    static struct timeval tstart;
    if(started == 0)
    {
        gettimeofday(&tstart, NULL);
        started = 1;
        return 0;
    }else
    {
        struct timeval now;
        gettimeofday(&now, NULL) ;
        return (now.tv_usec - tstart.tv_usec + (now.tv_sec - tstart.tv_sec) * 1000000)/1000;
    }
}
void getHumanReadableTime(float start)
{
    float all = ((float)getMilliSecond() - start)/1000 ;
    float allMs =  all * 1000.0f;
    int allS = (int)(allMs / 1000);
    int milliseconds = (((int)allMs) % 1000);
    int minutes = allS / 60;
    int seconds = allS % 60;
    std::cout
        <<"**************Welcome**************************\n"
        <<"[Total time]:\t\t" << allMs<< "ms\n"
        <<"[milliseconds]:\t\t" << milliseconds<< "ms\n"
        <<"[seconds]:\t\t" << seconds   << "s\n"
        <<"[minutes]:\t\t" << minutes  << "m\n"
        << std::endl;
}
int listFilesDirectory(vector<string>& images, const string& directory)
{
    DIR *dir;
    class dirent *ent;
    dir = opendir(directory.c_str());
    if(!dir)
    {
        std::cout << "ERROR:could not find directory in listFilesDirectory:\n"<<directory.c_str();
        return -1;
    }
    while((ent = readdir(dir))!= NULL )
    {
        if( ent->d_type == DT_REG)
        {
            string image(ent->d_name);            
            string ext = image.substr(image.rfind("."));
            if(!ext.compare(".jpg") || !ext.compare(".png"))
            {
                images.push_back(image);
            }
            else
            {
                std::cout << "WARNING:incorrect file no jpg nor png in listFilesDirectory:\n"<<image.c_str();
            }
        }
    }
    closedir(dir);
    return 0;
}
int main(int argc, char * argv[])
{
    char* buffer = (char*)malloc(1024);
    // Cambiar el flujo de salida hacia el buffer, una vez que se llene el buffer
    // de capacidad 1024 entonces dirigir el flujo de salida hacia el fichero fisico que es
    // el stdout.
    setvbuf( stdout , buffer, _IOLBF , 1024 );
    const string slash = "/";
    int blocks = 0;
    int svm = 0;
    double timings[5] = {0.0,0.0,0.0,0.0,0.0};
    try
    {
//    float start;
    string outputDir,directory,config;
    outputDir=argv[1];
    directory=argv[2];
    config=argv[3];
    //cout <<"out:: "<<outputDir<<" dir:: "<<directory<<" config:: "<<config<<endl;
    vector<string> images;
    vector<string> detections;
    if(listFilesDirectory(images, directory))
    {
        cout << "ERROR:finish Main"<<"\n" <<endl;
        return -1;

    }
    for(unsigned int i=0; i < images.size(); i++)
    {
        string outputPath = images[i].substr(images[i].rfind("/")+1).append(".detects");
        string imagesPath = directory + slash + images[i];
        string detectsPath = outputDir + slash + outputPath;
        detections.push_back(detectsPath);
        images[i] = imagesPath;
    }
    HOGPlanner hp;
    if(hp.readFromConfigFile(config))
    {
        std::cout << "ERROR:readFromConfigFile failed in Main.\n";
        return -1;
    }
    if(hp.readFromModelFile())
    {
        std::cout << "ERROR:readFromModelFile failed in Main.\n";
        return -1;
    }
//    start=(float)getMilliSecond();
    if(hp.detectInImages(images, detections, &blocks, &svm, timings) )
    {
        std::cout << "ERROR:detectInImages failed in Main.\n";
        return -1;
    }
//    getHumanReadableTime(start);
    printf("\nstatistics:\n");
    printf("Blocks:\t%d\n",blocks);
    printf("SVM:\t%d\n", svm);
    printf("gradients:\t %f\n", timings[0] / images.size());
    printf("blocks:\t\t %f\n", timings[1] / images.size());
    printf("svm:\t\t %f\n", timings[2] / images.size());
    printf("nms:\t\t %f\n", timings[3] / images.size());
    printf("\ntime per frame:\t %f\n\n", timings[4] / images.size());
    return 0;
    }
catch (std::exception const &exc)
{
    std::cerr << "Exception caught----------->" << exc.what() << "\n";
}
catch (...)
{
    std::cerr << "Unknown exception caught\n";
}

}


