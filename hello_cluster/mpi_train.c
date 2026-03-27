// Simple MPI program to emulate what the Excel spreadsheet model does for:
//
// velocity(t) = Sum[(accel(t)]
//
// position(t) = Sum[velocity(t)]
//
// Using as many ranks as you wish - should ideally divide into 1800 to avoid possible residual errors.
//
// Due to loop carried dependencies, this requires basic dynamic programming to use the final condition of one
// rank as the initial condition of others at a later time, but adjusted after the computations are done in parallel!
//
// This is identical to what's done in the OpenMP version of this code, but information is shared via message passing rather
// than shared memory.
//
// Note: this program will be helpful for the Ex #4 train problem where the anti-derivatives are unknown and all functions must
//       be integrated numerically.
//
// Sam Siewert, 2/24/2023, Cal State Chico - https://sites.google.com/mail.csuchico.edu/sbsiewert/home
//
// Please use as you see fit, but please do cite and retain a link to this original source
// here or my github (https://github.com/sbsiewertcsu/numeric-parallel-starter-code)
//

// included in "interpolation_functions.h"
//#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <omp.h>

#include "interpolation_functions.h"

#define DEBUG_TRACE
#define DEBUG_STEP (100)

// Function tables used in Ex #3 and #4 as well as test profiles for sine and a constant
// All funcitons have 1801 entries for time = 0.0 to 1800.0
#include "ex4.h"
//#include "ex3.h"
//#include "sine.h"
//#include "const.h"

const int MAX_STRING = 512;

int main(void)
{
    char resultmsg[MAX_STRING];
    char hostname[MAX_STRING];
    char nodename[MAX_STRING];
    //char nodename[MPI_MAX_PROCESSOR_NAME];
    int comm_sz;
    int my_rank, namelen, idx;

    // Parallel summing example
    double local_sum=0.0, g_sum=0.0;
    const int TABLELEN = sizeof(DefaultProfile)/sizeof(double);
    double default_sum[TABLELEN];
    double default_sum_of_sums[TABLELEN];
    //int residual;

    // Fill in default_sum array used to simulate a new table of values, such as
    // a velocity table derived by integrating acceleration
    //
    for(idx = 0; idx < TABLELEN; idx++)
    {
        default_sum[idx]=0.0;
        default_sum_of_sums[idx]=0.0;
    }


    printf("Will divide up work for input table of size = %d\n", TABLELEN);
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    const double DT = 0.001;
    // this is the range of table values a rank processes to
    const int SUBRANGE = TABLELEN / comm_sz;
    // this is the range of times the rank processes
    // TABLELEN - 1 == total_time
    // different from SUBRANGE!
    const double SUBRANGE_TIME = (TABLELEN - 1) / comm_sz;
    // this is how many steps each second is divide into
    const long STEPS_PER_SECOND = (long)(1.0 / DT);
    // this is how many steps the rank will go through
    const long STEPS_PER_RANK = (long)(SUBRANGE_TIME / DT);
    

    // For train, residual is one, but this is really just the first/last time index.
    // In general, this extra work would be handled by the last rank for simplicity, but
    // it is ignored here since we know 1801 table entries if for 1800 steps.
    //
    // This should be fixed for a more general solution or checked and user guided on correct input for
    // division of work.
    //
    //residual = TABLELEN % comm_sz;

    printf("Went parallel: rank %d of %d doing work for %d steps\n", my_rank, comm_sz, SUBRANGE);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // START PARALLEL PHASE 1: Sum original DefaultProfile LUT by rank
    //                         which is integration of accel(t) here.
    //
    if(my_rank != 0)
    {
        gethostname(hostname, MAX_STRING);
        MPI_Get_processor_name(nodename, &namelen);

        const double T_START = my_rank * SUBRANGE_TIME;
        double vel = 0.0;
        int table_idx = my_rank * SUBRANGE;
        // Now sum up the values in the LUT function
        for (long i = 0; i < STEPS_PER_RANK; i++) {
            double t = T_START + i * DT;
            vel += faccel(t) * DT;

            if ((i + 1) % STEPS_PER_SECOND == 0) {
                default_sum[table_idx] = vel;
                table_idx++;
            }
        }
        local_sum = vel;

        sprintf(resultmsg, "Sum of DefaultProfile for rank %d of %d on %s is %lf", my_rank, comm_sz, nodename, local_sum);
        MPI_Send(resultmsg, strlen(resultmsg)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        gethostname(hostname, MAX_STRING);
        MPI_Get_processor_name(nodename, &namelen);

        double T_START = my_rank * SUBRANGE_TIME;
        double vel = 0.0;
        int table_idx = 0;
        // Now sum up the values in the LUT function
        for (long i = 0; i < STEPS_PER_RANK; i++) {
            double t = T_START + i * DT;
            vel += faccel(t) * DT;

            if ((i + 1) % STEPS_PER_SECOND == 0) {
                default_sum[table_idx] = vel;
                table_idx++;
            }
        }
        local_sum = vel;

        printf("Sum of DefaultProfile for rank %d of %d on %s is %lf\n", my_rank, comm_sz, nodename, local_sum);

        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(resultmsg, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", resultmsg);
        }
    }

    // This should be the summation of DefaultProfile, which should match the spreadsheet for a train profile for DT=1.0
    MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank == 0) printf("\nRank 0 g_sum = %lf\n", g_sum);

    MPI_Barrier(MPI_COMM_WORLD);

    // DISTRIBUTE RESULTS: Now to correct and overwrite all default_sum, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data
    if(my_rank != 0)
    {
        MPI_Send(&default_sum[my_rank*SUBRANGE], SUBRANGE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&default_sum[q*SUBRANGE], SUBRANGE, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ADJUST FOR INITIAL CONDITION: **** SUPER IMPORTANT to adjust initial condition offset ****
            //
            // Adjust so initial condition is ending conditon of prior sum
            //
            // Good candidate for OpenMP parallel loop for MPI+OpenMP
            //
            // #pragma
            for(idx = q*SUBRANGE; idx < (q*SUBRANGE)+SUBRANGE; idx++)
                default_sum[idx] += default_sum[((q-1)*SUBRANGE)+SUBRANGE-1];
        }
    }
    // Make sure all ranks have the full new default table - overkill, so optimize by sending just what they need
    default_sum[TABLELEN-1]=default_sum[TABLELEN-2]; // fixes issue where last table entry is otherwise zero
    MPI_Bcast(&default_sum[0], TABLELEN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    //
    // END PARALLEL PHASE 1: Every rank has the same updated default_sum table now
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // TRACE: Just a double check on the FIRST MPI_Reduce and trace output of first phase
    if(my_rank == 0)
    {
#ifdef DEBUG_TRACE
        // Now rank zero has the data from each of the other ranks in one new table
        printf("\nVelocity TRACE: Rank %d sum of default_sum = %lf\n", my_rank, g_sum);
        for(idx = 0; idx < TABLELEN; idx+=DEBUG_STEP)
        {
            printf("Curr t=%d: a=%lf for v=%lf\n", idx, DefaultProfile[idx], default_sum[idx]);
        }
#endif
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // START PARALLEL PHASE 2: Now that all ranks have the new default_sum table, we can proceed to sum all of those sums as before
    //
    // Do the next round of summing from the new table, which is velocity(t) here, at same resolution
    // as the original accel(t) table with 1801 data points for time=0, ..., 1800.
    //

    // this is basically the same value of TABLELEN but as a
    // different type.
    const long unsigned int TSIZE = sizeof(default_sum) / sizeof(double);

    local_sum=0;

    if(my_rank != 0)
    {
        // Now sum up the values in the new LUT function default_sum
        const double T_START = my_rank * SUBRANGE_TIME;
        double dist = 0.0;
        int table_idx = my_rank * SUBRANGE;
        for (long i = 0; i < (long)(STEPS_PER_RANK); i++) {
            double t = T_START + i * DT;
            // default_sum == VelProfile
            dist += fvel(t, default_sum, &TSIZE) * DT;
            if ((i + 1) % STEPS_PER_SECOND == 0) {
                default_sum_of_sums[table_idx] = dist;
                table_idx++;
            }
        }
        local_sum = dist;
    }
    else
    {
        // Now sum up the values in the new LUT function default_sum
        const double T_START = my_rank * SUBRANGE_TIME;
        double dist = 0.0;
        int table_idx = 0;
        for (long i = 0; i < (long)(STEPS_PER_RANK); i++) {
            double t = T_START + i * DT;
            // default_sum == VelProfile
            dist += fvel(t, default_sum, &TSIZE) * DT;
            if ((i + 1) % STEPS_PER_SECOND == 0) {
                default_sum_of_sums[table_idx] = dist;
                table_idx++;
            }
        }
        local_sum = dist;
    }

    // This should be the summation of the sums, which should match the spreadsheet for a train profile for DT=1.0
    MPI_Reduce(&local_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank == 0) printf("\nRank 0 g_sum = %lf\n", g_sum);

    // DISTRIBUTE: Finally correct and overwrite all default_sum_of_sums, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data
    if(my_rank != 0)
    {
        MPI_Send(&default_sum_of_sums[my_rank*SUBRANGE], SUBRANGE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&default_sum_of_sums[q*SUBRANGE], SUBRANGE, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ADJUST FOR INITIAL CONDITION: **** SUPER IMPORTANT to adjust initial condition offset ****
            //
            // Adjust so initial condition is ending conditon of prior sum
            //
            // Good candidate for OpenMP parallel loop for MPI+OpenMP
            //
            for(idx = q*SUBRANGE; idx < (q*SUBRANGE)+SUBRANGE; idx++)
                default_sum_of_sums[idx] += default_sum_of_sums[((q-1)*SUBRANGE)+SUBRANGE-1];
        }
    }
    // Make sure all ranks have the full new default table - overkill, so optimize by sending just what they need
    default_sum_of_sums[TABLELEN-1]=default_sum_of_sums[TABLELEN-2]; // fixes issue where last table entry is otherwise zero
    MPI_Bcast(&default_sum_of_sums[0], TABLELEN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    //
    // END PHASE 2: Every rank has the same updated default_sum_of_sums table now
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // TRACE: Final double check on the SECOND MPI_Reduce and trace output of second phase
    if(my_rank == 0)
    {
#ifdef DEBUG_TRACE
        // Now rank zero has the data from each of the other ranks in one new table
        printf("\nPosition TRACE: Rank %d sum of default_sum = %lf\n", my_rank, g_sum);
        for(idx = 0; idx < TABLELEN; idx+=DEBUG_STEP)
        {
            printf("Curr t=%d: a=%lf for p=%lf\n", idx, DefaultProfile[idx], default_sum_of_sums[idx]);
        }
#endif
    }

    MPI_Finalize();
    return 0;
}

