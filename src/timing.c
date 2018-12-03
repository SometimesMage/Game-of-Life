#include "timing.h"
#include <sys/time.h>
#include <stdlib.h>

double currentTime()
{
   struct timeval now;
   gettimeofday(&now, NULL);
   
   return (double)now.tv_sec + (double)now.tv_usec / 1000000.0;
}