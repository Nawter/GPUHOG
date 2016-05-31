#include <math.h>
#include <stdio.h>
#include "SkipSupperssion.h"
#include "Global.h"
#include "GlobalParameters.h"
#include "HOGPlanner.h"
#include "DumpDetections.h"
#include <iostream>
#define NMS_MAX_ITERATIONS	100
#define NMS_MODE_EPSILON	((float)1E-5f)
#define NMS_FINAL_DIST		1.f
namespace UC3M_HOG_GPU
{
    const float sigmaFactor = 4.f;
    float sigmaX;
    float sigmaY;
    float sigmaS;
    /* Inline functions are not always important, but it is good to understand them.
     * The basic idea is to save time at a cost in space. Inline functions are a lot like a placeholder.
     * Once you define an inline function, using the 'inline' keyword,
     * whenever you call that function the compiler will replace the function call
     * with the actual code from the function.*/
    inline float distance(nmsDetection* p1, nmsDetection* p2)
    {
        const float expScale = expf(p2->scale);
        float ns[3];
        ns[0] = sigmaX * expScale;
        ns[1] = sigmaY * expScale;
        ns[2] = sigmaS;      
        float b[3];
        b[0] = (p2->x - p1->x) / ns[0];
        b[1] = (p2->y - p1->y) / ns[1];
        b[2] = (p2->scale - p1->scale) / ns[2];        
        return b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
    }
    void NMS::processDetectionList(detectionList& detection, detectionList& detectionsNMS, LocalParameters& params)
    {

        const int windowWidth = params.WINDOW_WIDTH;
        const int windowHeight = params.WINDOW_HEIGHT;
        sigmaX = SIGMA_FACTOR;
        sigmaY = (params.WINDOW_HEIGHT / (float)params.WINDOW_WIDTH) * sigmaFactor;
        sigmaS = logf(SIGMA_SCALE);       
        nmsDetection nms[4096];
        float nmsScore[4096];
        nmsDetection modes[4096];
        nmsDetection ms[4096];
        // [Zaid-11] cambio el dia 25/10/2014
//        std::cout << "---SkipSuppression---processDetectionList---detection--->"<< detection.size() << std::endl;
        // Fin [Zaid-11]
        if( detection.size() > 4096 )
        {
            printf("too many detections!\n");
            return;
        }
        int cont = 0;
        for(size_t i=0; i < detection.size(); i++)
        {
            Detection* ptr = &(detection[i]);
            nms[cont].x = ptr->x + floorf((windowWidth*ptr->scale)) / 2.f;
            nms[cont].y = ptr->y + floorf((windowHeight*ptr->scale)) / 2.f;
            nms[cont].scale = logf(ptr->scale);
            nmsScore[cont] = fmax( ptr->score, 0.f );            
            cont++;
        };
        float nX;
        float nY;
        float nZ;
        float pI[3];
        float pJ[3];
        float numer[3];
        float denum[3];
        for(int i=0; i < cont; i++)
        {
            numer[0] = 0.f;
            numer[1] = 0.f;
            numer[2] = 0.f;
            denum[0] = 0.f;
            denum[1] = 0.f;
            denum[2] = 0.f;
            for(int j=0; j < cont; j++)
            {
                float w;
                const float expScale = expf(nms[j].scale);
                nX = sigmaX * expScale;
                nY = sigmaY * expScale;
                nZ = sigmaS;
                pI[0] = nms[i].x / nX;
                pI[1] = nms[i].y / nY;
                pI[2] = nms[i].scale / sigmaS;
                pJ[0] = nms[j].x / nX;
                pJ[1] = nms[j].y / nY;
                pJ[2] = nms[j].scale / sigmaS;
                float sqrdist = (pI[0] - pJ[0]) * (pI[0] - pJ[0]) +
                                (pI[1] - pJ[1]) * (pI[1] - pJ[1]) +
                                (pI[2] - pJ[2]) * (pI[2] - pJ[2]);
                w = nmsScore[j] * expf(-sqrdist/2.f) / sqrtf( nX * nY * nZ);
                numer[0] += w * pJ[0];
                numer[1] += w * pJ[1];
                numer[2] += w * pJ[2];
                denum[0] += w / nX;
                denum[1] += w / nY;
                denum[2] += w / nZ;                
            }
            ms[i].x = numer[0] / denum[0];
            ms[i].y = numer[1] / denum[1];
            ms[i].scale = numer[2] / denum[2];           
        }
        for(int i=0; i < cont; i++)
        {
            nmsDetection point;
            nmsDetection movedPoint;
            movedPoint.x = ms[i].x;
            movedPoint.y = ms[i].y;
            movedPoint.scale = ms[i].scale;        
            int count = 0;
            do
            {
                point.x = movedPoint.x;
                point.y = movedPoint.y;
                point.scale = movedPoint.scale;
                float n[3] = {0,0,0};
                float d[3] = {0,0,0};               
                for(int j=0; j < cont; j++)
                {
                    float w;
                    const float exp_scale = expf(nms[j].scale);
                    nX = sigmaX * exp_scale;
                    nY = sigmaY * exp_scale;
                    nZ = sigmaS;
                    pI[0] = point.x / nX;
                    pI[1] = point.y / nY;
                    pI[2] = point.scale / sigmaS;
                    pJ[0] = nms[j].x / nX;
                    pJ[1] = nms[j].y / nY;
                    pJ[2] = nms[j].scale / sigmaS;
                    float sqrdist = (pI[0] - pJ[0]) * (pI[0] - pJ[0]) +
                                    (pI[1] - pJ[1]) * (pI[1] - pJ[1]) +
                                    (pI[2] - pJ[2]) * (pI[2] - pJ[2]);
                    w = nmsScore[j] * expf(-sqrdist/2.f) / sqrtf( nX * nY * nZ);
                    n[0] += w * pJ[0];
                    n[1] += w * pJ[1];
                    n[2] += w * pJ[2];
                    d[0] += w / nX;
                    d[1] += w / nY;
                    d[2] += w / nZ;
                }
                movedPoint.x = n[0] / d[0];
                movedPoint.y = n[1] / d[1];
                movedPoint.scale = n[2] / d[2];
                count++;
            } while(  ( count < NMS_MAX_ITERATIONS ) && ( distance(&point, &movedPoint) > NMS_MODE_EPSILON ) );
            modes[i].x = movedPoint.x;
            modes[i].y = movedPoint.y;
            modes[i].scale = movedPoint.scale;
            modes[i].score = nmsScore[i];
        }
        nmsDetection nmsModes[4096];
        int nValidModes =0;
        // jugar con esos valores y ver los cambios sobre las imagenes y su deteccion.
        // extract the valid modes from modes array (output
    #if NMS_MAXIMUM_SCORE == 0
        for(int i=0; i < count; i++)
        {
            int include = 1;
            for(int j=0; j < nValidModes; j++)
            {
                if( distance( &(nmsModes[j]), &(modes[i]) ) < NMS_FINAL_DIST)
                {
                    include = 0;
                    break;
                }
            }
            if( include )
            {
                nmsModes[nValidModes].x = modes[i].x;
                nmsModes[nValidModes].y = modes[i].y;
                nmsModes[nValidModes].scale = modes[i].scale;
                nValidModes++;
            }
        }
        // find score for each valid mode
        for(int i=0; i < nValidModes; i++)
        {
            float average = 0.f;
            for(int j=0; j < count; j++)
            {
                const float expScale = expf(nms[j].scale);
                nX = sigmaX * expScale;
                nY = sigmaY * expScale;
                nZ = sigmaS;
                float p[3];
                p[0] = (nms[j].x - nmsModes[i].x) / nX;
                p[1] = (nms[j].y - nmsModes[i].y) / nY;
                p[2] = (nms[j].scale - nmsModes[i].scale) / nZ;
                float sqrDist = p[0]*p[0] +  p[1]*p[1] + p[2]*p[2];
                average += nms_score[j] * expf(-sqrDist/2.f)/sqrtf(nX * nY * nZ);
            }
            float scale = expf(nmsModes[i].scale);
            int x = (int) ceilf( nmsModes[i].x - window_width*scale / 2.f );
            int y = (int) ceilf( nmsModes[i].y - window_height*scale / 2.f );         
            detectionsNMS.push_back(Detection(x,y,scale,average));
        }
    #else
        for(int i=0; i < cont; i++)
        {
            int include = 1;
            for(int j=0; j < nValidModes; j++)
            {
                nmsDetection *p1 = &(nmsModes[j]);
                nmsDetection *p2 = &(modes[i]);
                    const float exp_scale = (expf(p2->scale) + expf(p1->scale) ) / 2.f;
                    float ns[3];
                    ns[0] = sigmaX * exp_scale;
                    ns[1] = sigmaY * exp_scale;
                    ns[2] = sigmaS;
                    float b[3];
                    b[0] = (p2->x - p1->x) / ns[0];
                    b[1] = (p2->y - p1->y) / ns[1];
                    b[2] = (p2->scale - p1->scale) / ns[2];
                    float dist = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
                if(dist <= NMS_FINAL_DIST)
                {
                    include = 0;
                    if( nmsModes[j].score < modes[i].score )
                    {
                        nmsModes[j].score = modes[i].score;
                    }
                    break;
                }
            }
            if( include )
            {
                nmsModes[nValidModes].x = modes[i].x;
                nmsModes[nValidModes].y = modes[i].y;
                nmsModes[nValidModes].scale = modes[i].scale;
                nmsModes[nValidModes].score = modes[i].score;
                nValidModes++;
            }
        }
        for(int i=0; i < nValidModes; i++)
        {
            float scale = expf(nmsModes[i].scale);
            int x = (int) ceilf( nmsModes[i].x - windowWidth*scale / 2.f );
            int y = (int) ceilf( nmsModes[i].y - windowHeight*scale / 2.f );         
            detectionsNMS.push_back(Detection(x, y, scale, nmsModes[i].score));
        }
    #endif
    }

}
