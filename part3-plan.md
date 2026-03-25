# Part 3 Outline: Single MPI Train Simulation Program

## Context
Part 3 requires integrating a train's acceleration profile (A(t)) twice numerically — once to get velocity V(t), once to get position X(t) — using MPI parallelism. The challenge is a **loop-carried dependency**: each time step's velocity depends on the previous step's velocity, making naive parallelism impossible. The assignment points to `compare.c` as the blueprint for solving this with MPI.

---

## Files to Bring Together

| File | What you take from it |
|------|----------------------|
| `hello_cluster/compare.c` | The entire MPI structure (2-phase parallel integration pattern) |
| `functiongen/timeprofiles_omp.c` | `faccel()` linear interpolation, `table_accel()`, `fvel()`, `table_vel()`, timing code, sequential Riemann sum loop |
| `functiongen/ex4.h` | `DefaultProfile[]` — the 1801-entry acceleration lookup table (t=0 to t=1800, 1 sec apart) |

---

## The Core Problem: Loop-Carried Dependency

Sequential integration looks like:
```
vel[0] = 0.0
for each timestep t:
    vel[t] = vel[t-1] + accel(t) * dt   // depends on previous vel
    pos[t] = pos[t-1] + vel[t] * dt     // depends on previous pos
```

You can't split this across ranks naively. `compare.c` solves it with **dynamic programming / offset correction**:
1. Each rank integrates its subrange independently, starting from local_sum=0
2. Rank 0 collects all partial results and **adds the ending value of rank N-1 to all of rank N's values** (the initial condition correction)
3. Broadcast the corrected table to all ranks
4. Repeat for position using the corrected velocity table

---

## Step-by-Step Build Plan

### Step 1: Start with `compare.c` as your base file
Copy it to a new file, e.g. `train_mpi.c`. This gives you:
- `MPI_Init`, `MPI_Comm_size`, `MPI_Comm_rank`
- The 2-phase parallel pattern (Phase 1 = accel→vel, Phase 2 = vel→pos)
- `MPI_Send`/`MPI_Recv` for collecting partial arrays from each rank
- The **critical offset correction loop** (lines 150-151 and 226-227 in compare.c)
- `MPI_Bcast` to share completed table with all ranks
- `MPI_Barrier` and `MPI_Reduce` for synchronization and verification

### Step 2: Add the interpolation functions from `timeprofiles_omp.c`
Copy these functions verbatim into your new file:
- `table_accel(int timeidx)` — bounds-checked lookup into `DefaultProfile[]`
- `table_vel(int timeidx)` — bounds-checked lookup into `VelProfile[]`
- `faccel(double time)` — linear interpolation for A(t) at any floating-point time
- `fvel(double time)` — linear interpolation for V(t) at any floating-point time

Also add the include: `#include "ex4.h"` (already in compare.c and timeprofiles_omp.c)

### Step 3: Replace the coarse dt=1.0 loop with a fine-grained Riemann loop

`compare.c` works at dt=1.0 (one step per table entry, 1801 steps). You need dt=0.001 or dt=0.0001.

Change the inner loop from:
```c
// compare.c style (dt=1.0):
for(idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
{
    local_sum += DefaultProfile[idx];   // crude, dt=1.0
    default_sum[idx] = local_sum;
}
```

To a fine-grained loop using `faccel()` (from timeprofiles_omp.c):
```c
// dt is set from command line or hardcoded (0.001 or 0.0001)
double dt = 0.001;
double total_time = 1800.0;
double subrange_time = total_time / comm_sz;  // e.g. 450.0 sec per rank with 4 ranks

double t_start = my_rank * subrange_time;
double t_end   = t_start + subrange_time;
long my_steps = (long)(subrange_time / dt);   // 450,000 steps per rank at dt=0.001
double vel = 0.0;

for(long i = 0; i < my_steps; i++)
{
    double t = t_start + i * dt;
    vel += faccel(t) * dt;   // Left Riemann sum; faccel() interpolates from 1-sec table
    // periodically store vel into VelProfile[] at the right table index
}
```

The key insight: each rank owns a time-slice `[my_rank * T/N, (my_rank+1) * T/N]` where T=1800.
`dt` controls the step granularity. `faccel(t)` uses linear interpolation to look up acceleration
at any fractional time from the 1-second-spaced DefaultProfile[] table.

### Step 4: Keep the offset correction exactly as in `compare.c`
All ranks run in parallel with NO waiting. Each starts from local_sum=0 (wrong initial condition).
After all ranks finish, rank 0 collects partial arrays via `MPI_Recv` and applies the correction:
```c
for(int q = 1; q < comm_sz; q++)
{
    MPI_Recv(&VelProfile[q*subrange], subrange, MPI_DOUBLE, q, 0, ...);
    // Add ending velocity of rank q-1 to ALL of rank q's values
    for(idx = q*subrange; idx < (q*subrange)+subrange; idx++)
        VelProfile[idx] += VelProfile[((q-1)*subrange)+subrange-1];
}
```
Example with 4 ranks:
- Rank 0 ends at vel=50 → add 50 to all of Rank 1's values
- Rank 1 (corrected) ends at vel=90 → add 90 to all of Rank 2's values
- Rank 2 (corrected) ends at vel=100 → add 100 to all of Rank 3's values
This works because integration is additive — a wrong starting point is fixed by a constant offset.

**Note on OpenMP hybrid:** The outer correction loop (over ranks) is inherently sequential, but the
inner loop (adding offset to one rank's entries) could use `#pragma omp parallel for`. compare.c
mentions this at line 148. Worth mentioning in the report, though speedup is minimal since the
correction is O(table_size) vs O(millions) for the integration itself.

### Step 5: Add timing from `timeprofiles_omp.c`
```c
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
// ... integration ...
clock_gettime(CLOCK_MONOTONIC, &end);
double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;
```
Wrap rank 0's main work in the timer and report elapsed time for dt=0.001 and dt=0.0001.

### Step 6: Output final results from rank 0
Print final velocity (index 1800) and final position (index 1800), and verify:
- Peak velocity ≈ 320 km/hr (≈ 88.9 m/s)
- Final position ≈ 122,000 m (±100 m)
- Velocity at t=0 and t=1800 ≈ 0.0

---

## What Each File Contributes (Visual Summary)

```
compare.c                    timeprofiles_omp.c          ex4.h
-----------                  ------------------          -----
MPI_Init/Finalize       +    faccel(time)            +   DefaultProfile[]
subrange division            fvel(time)                  (1801 accel values)
Phase 1 loop skeleton        table_accel(timeidx)
offset correction            table_vel(timeidx)
MPI_Send/Recv/Bcast          Riemann sum inner loop
Phase 2 loop skeleton        timing code (clock_gettime)
MPI_Reduce (verify)          VelProfile[], PosProfile[]
```

---

## Verification

- **Part a** (dt=0.001): Run sequentially (1 process), report final vel and pos
- **Part b** (dt=0.0001): Run sequentially, compare final vel/pos to part a, time both
- **Part c** (scaling): Run with 2, 4, 8, 16 MPI processes using `mpirun -n N ./train_mpi`, plot time vs N

---

## Key Gotcha (from compare.c comments)
The table has 1801 entries for t=0..1800. When dividing by N ranks, `1801 % N` may leave a residual. `compare.c` ignores this (it works for N that divides evenly into 1800). For a general solution, have the last rank handle the residual steps.
