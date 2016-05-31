#include <iostream>
#include <fstream>
#include <cassert>
#include <climits>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include "Global.h"
#include "GlobalParameters.h"
#include "LocalParameters.h"
namespace UC3M_HOG_GPU
{
    int GlobalParameters::getMinDescriptorHeight()
    {
        assert(configModels.size() > 0);
        int min= INT_MAX;
        for(size_t i=0; i<configModels.size();i++)
        {
            min = std::min(configModels[i].DESCRIPTOR_HEIGHT,min);
        }
        return min;
    }
    int GlobalParameters::getMinDescriptorWidth()
    {
        assert(configModels.size() > 0);
        int min= INT_MAX;
        for(size_t i=0; i<configModels.size();i++)
        {
            min = std::min(configModels[i].DESCRIPTOR_WIDTH,min);
        }
        return min;
    }
    int GlobalParameters::computeMinWindowHeight()
    {
        assert(configModels.size() > 0);
        int min= INT_MAX;
        for(size_t i=0; i<configModels.size();i++)
        {
            min = std::min(configModels[i].WINDOW_HEIGHT,min);
        }
        return min;
    }
    int GlobalParameters::computeMinWindowWidth()
    {
        assert(configModels.size() > 0);
        int min= INT_MAX;
        for(size_t i=0; i<configModels.size();i++)
        {
            min = std::min(configModels[i].WINDOW_WIDTH,min);
        }
        return min;
    }
    int GlobalParameters::computeMaxWindowHeight()
    {
        assert(configModels.size() > 0);
        int max= -INT_MAX;
        for(size_t i=0; i<configModels.size();i++)
        {
            max = std::max(configModels[i].WINDOW_HEIGHT,max);
        }
        return max;
    }
    int GlobalParameters::computeMaxWindowWidth()
    {
        assert(configModels.size() > 0);
        int max= -INT_MAX;
        for(size_t i=0; i<configModels.size();i++)
        {
            max = std::max(configModels[i].WINDOW_WIDTH,max);
        }
        return max;
    }
    void extractPath(std::string& string, std::string& location)
    {
        int last = string.rfind('/');
        if (last <= 0)
        {
            location=string;
        }
        else
        {
            location=string.substr(0,last);
        }
    }
    void trimString(std::string& string)
    {
        if(!string.empty())
        {
            int first = string.find_first_not_of(" \t");
            int last  = string.find_last_not_of(" \t");
            std::string aux = string.substr(first, last - first + 1);
            string.erase();
            string = aux;

        }
    }
    int GlobalParameters::loadFromFile(std::string& file)
    {
        //std::cout<< "Reading Parameters from file:"<<file.c_str()<<"\n";
        extractPath(file, location);
        std::ifstream inputStream(file.c_str());       
        if(!inputStream.good())
        {
            std::cout<< "ERROR:failed to open parameters file in loadFromFile:"<<file.c_str()<<"\n";
            return -1;
        }
        char c;
        while(inputStream.good())
        {
            inputStream.read(&c, 1);
            char name[128];
            char comment[256];
            if( c == '[' )
            {
                //std::cout<< "paso uno:"<<"\n";
                inputStream.getline(name, 128, ']');
                LocalParameters configModel;
                configModel.id = std::string(name);
                //printf("new ModelParameters object: %s\n", configModel.id.c_str());
                configModels.push_back(configModel);
            }
            else if( c == '#' )
            {
                //std::cout<< "paso else-1:"<<"\n";
                inputStream.getline(comment, 256);
            }
            else
            {
//                std::cout<< "paso else-2:"<<"\n";
                inputStream.putback(c);
                if( !strnlen(name, 128) || !strncmp(name, "global", 128) )
                {
                    std::string line;
                    std::getline(inputStream, line);
                    std::istringstream stringStream(line);
                    char aux[128];
                    stringStream.getline(aux, 128, '=');
                    std::string key(aux);
                    trimString(key);
                    stringStream.getline(aux,128);
                    std::string value(aux);
                    trimString(value);
//                    std::cout<< "key:"<<key.c_str()<<"\n";
//                    std::cout<< "value:"<<value.c_str()<<"\n";
////                    if(key.empty() )
//                    {
//                        continue;
//                    }
                    if(!key.empty())
                    {
                        if( value.empty() )
                        {
                            std::cout<< "ERROR: empty value for key in loadFromFile:\n"<<key.c_str();
                            return -1;
                        }
                        std::cout<< "WARNING:currently global parameters have no effect in loadFromFile:\n";
                    }
                }
                else
                {
                    std::string line;
                    getline(inputStream, line);
                    std::istringstream stringStream(line);
                    char aux[128];
                    stringStream.getline(aux, 128, '=');
                    std::string key(aux);
                    trimString(key);
                    stringStream.getline(aux,128);
                    std::string value(aux);
                    trimString(value);
//                    std::cout<< "key:"<<key.c_str()<<"\n";
//                    std::cout<< "value:"<<value.c_str()<<"\n";
//                    if( key.empty() )
//                    {
//                        continue;
//                    }
                    if(!key.empty())
                    {
                        if( value.empty() )
                        {
                            std::cout<< "ERROR: empty value for key in loadFromFile:\n"<<key.c_str();
                            return -1;
                        }
                        if (configModels.size()==0)
                        {
                            std::cout<< "ERROR: There is no model in loadFromFile:\n";
                            return -1;
                        }
                        assert(configModels.size() > 0 );
                        //std::cout<< "size:"<<configModels.size()<<"\n";
                        LocalParameters &configModel = configModels[configModels.size()-1];
                        if( !key.compare("DESCRIPTOR_HEIGHT") )
                        {
                            configModel.DESCRIPTOR_HEIGHT = atoi(value.c_str());
                        }
                        else if( !key.compare("DESCRIPTOR_WIDTH") )
                        {
                            configModel.DESCRIPTOR_WIDTH = atoi(value.c_str());
                        }
                        else if( !key.compare("WINDOW_HEIGHT") )
                        {
                            configModel.WINDOW_HEIGHT = atoi(value.c_str());
                        }
                        else if( !key.compare("WINDOW_WIDTH") )
                        {
                            configModel.WINDOW_WIDTH = atoi(value.c_str());
                        }
                        else if( !key.compare("FILE") )
                        {
                            configModel.filename = value;
                        }
                        else if( !key.compare("MIN_SCALE") )
                        {
                            configModel.minScale = atof(value.c_str());
                        }
                        else if( !key.compare("MAX_SCALE") )
                        {
                            configModel.maxScale = atof(value.c_str());
                        }
                        else
                        {
                            std::cout<< "ERROR:unknown key value pair\t"<<"section:"<<name<<"key:"<<key.c_str()
                                     <<"value:"<<value.c_str()<<"in loadFromFile"<<"/n";
                            return -2;
                        }
                    }
                }
            }
        }
        return 0;
    }
}
