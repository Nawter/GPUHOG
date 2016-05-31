#include <sys/time.h>
typedef struct
{
    timespec start;
    timespec end;
} Timer;
// NOTE: you need to link with -lrt when using this (on Linux / Unix)
extern void startTimer(Timer*);
extern void stopTimer(Timer*);
extern double getTimer(Timer*);
