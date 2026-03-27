#include "interpolation_functions.h"

double table_accel(const int TIMEIDX)
{
    const long unsigned int TSIZE = sizeof(DefaultProfile) / sizeof(double);

    // Check array bounds for look-up table
    if(TIMEIDX > TSIZE)
    {
        printf("timeidx=%d exceeds table size = %lu and range %d to %lu\n", TIMEIDX, TSIZE, 0, TSIZE-1);
        exit(-1);
    }

    return DefaultProfile[TIMEIDX];
}

double table_vel(const int TIMEIDX, const double VELPROFILE[], const long unsigned int* TSIZE)
{
    if(TIMEIDX > *TSIZE)
    {
        printf("timeidx=%d exceeds table size = %ld and range %d to %ld\n", TIMEIDX, (*TSIZE), 0, (*TSIZE)-1);
        exit(-1);
    }

    return VELPROFILE[TIMEIDX];
}

double faccel(const double TIME)
{
    // The timeidx is an index into the known acceleration profile at a time <= time of interest passed in
    //
    // Note that conversion to integer truncates double to next lowest integer value or floor(time)
    //
    const int TIMEIDX = (int)TIME;

    // The timeidx_next is an index into the known acceleration profile at a time > time of interest passed in
    //
    // Note that the conversion to integer truncates double and the +1 is added for ceiling(time)
    //
    const int TIMEIDX_NEXT = ((int)TIME)+1;

    // delta_t = time of interest - time at known value < time
    //
    // For more general case
    // double delta_t = (time - (double)((int)time)) / ((double)(timeidx_next - timeidx);
    //
    // If time in table is always 1 second apart, then we can simplify since (timeidx_next - timeidx) = 1.0 by definition here
    const double DELTA_T = TIME - (double)((int)TIME);

    return ( 
               // The accel[time] is a linear value between accel[timeidx] and accel[timeidx_next]
               // 
               // The accel[time] is a value that can be determined by the slope of the interval and accel[timedix] 
               //
               // I.e. accel[time] = accel[timeidx] + ( (accel[timeidx_next] - accel[timeidx]) / ((double)(timeidx_next - timeidx)) ) * delta_t
               //
               //      ((double)(timeidx_next - timeidx)) = 1.0
               // 
               //      accel[time] = accel[timeidx] + (accel[timeidx_next] - accel[timeidx]) * delta_t
               //
               table_accel(TIMEIDX) + ( (table_accel(TIMEIDX_NEXT) - table_accel(TIMEIDX)) * DELTA_T)
           );
}

double fvel(const double TIME, const double VELPROFILE[], const long unsigned int* TSIZE)
{
    const int TIMEIDX = (int)TIME;
    const int TIMEIDX_NEXT = ((int)TIME)+1;
    const double DELTA_T = TIME - (double)((int)TIME);

    return (table_vel(TIMEIDX, VELPROFILE, TSIZE) + ( (table_vel(TIMEIDX_NEXT, VELPROFILE, TSIZE) - table_vel(TIMEIDX, VELPROFILE, TSIZE)) * DELTA_T) );
}
