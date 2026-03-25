extern double DefaultProfile[1801];

double VelProfile[sizeof(DefaultProfile) / sizeof(double)];

// bounds-checked lookup into DefaultProfile[]
double table_accel(int timeidx);
// bounds-checked lookup into VelProfile[]
double table_vel(int timeidx);

// Simple linear interpolation example for table_accel(t) for any floating point t value
// for a table of accelerations that are 1 second apart in time, evenly spaced in time.
//
// accel[timeidx] <= accel[time] < accel[timeidx_next]
//
//
double faccel(double time);
// linear interpolation for V(t) at any floating-point time
double fvel(double time);
