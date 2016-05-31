#include "Timer.h"
#include <time.h>

void startTimer(Timer* t)
{
    clock_gettime(CLOCK_MONOTONIC, &((*t).start));
}

void stopTimer(Timer* t)
{
    clock_gettime(CLOCK_MONOTONIC, &((*t).end));
}

// return timer count in ms
double getTimer(Timer* t)
{
    timespec temp;
    if (((*t).end.tv_nsec - (*t).start.tv_nsec) < 0)
        temp.tv_nsec = (*t).end.tv_nsec - (*t).start.tv_nsec + 1000000000;
    else
        temp.tv_nsec = (*t).end.tv_nsec - (*t).start.tv_nsec;
    return temp.tv_nsec / 1000000.;
}
