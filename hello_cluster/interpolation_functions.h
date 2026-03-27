#include <stdio.h>
#include <stdlib.h>

extern double DefaultProfile[1801];

// bounds-checked lookup into DefaultProfile[]
double table_accel(const int TIMEIDX);
// bounds-checked lookup into VelProfile[]
double table_vel(const int TIMEIDX, const double VELPROFILE[], const long unsigned int* TSIZE);

// Simple linear interpolation example for table_accel(t) for any floating point t value
// for a table of accelerations that are 1 second apart in time, evenly spaced in time.
//
// accel[timeidx] <= accel[time] < accel[timeidx_next]
//
//
double faccel(const double TIME);
// linear interpolation for V(t) at any floating-point time
double fvel(const double TIME, const double VELPROFILE[], const long unsigned int* TSIZE);
