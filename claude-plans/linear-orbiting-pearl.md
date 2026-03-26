# Plan: Finish converting all loops to fine-grained Riemann sums

## Context
`mpi_train.c` is partially converted. The non-rank-0 branch of Phase 1 already uses `faccel(t) * dt` interpolation. But rank 0's Phase 1 loop and **both** Phase 2 loops still use the old dt=1.0 code. We need to finish the conversion.

## Current State of Each Loop

| Loop | Location | Status |
|------|----------|--------|
| Phase 1, ranks 1+ | Lines 102-114 | **Already converted** to `faccel(t)*dt` |
| Phase 1, rank 0 | Lines 126-130 | **Still old code:** `local_sum += DefaultProfile[idx]` |
| Phase 1 correction | Lines 166-167 | No change needed |
| Phase 2, ranks 1+ | Lines 204-208 | **Still old code:** `local_sum += default_sum[idx]` |
| Phase 2, rank 0 | Lines 213-217 | **Still old code:** `local_sum += default_sum[idx]` |
| Phase 2 correction | Lines 242-243 | No change needed |

## Bug Fix
Line 62: `double total_time = sizeof(DefaultProfile) - 1;` — `sizeof` returns bytes, not element count. Fix to:
```c
double total_time = (sizeof(DefaultProfile)/sizeof(double)) - 1;  // 1800.0
```

## Changes

### 1. Fix `total_time` (line 62)
```c
double total_time = (sizeof(DefaultProfile)/sizeof(double)) - 1;
```

### 2. Convert rank 0's Phase 1 loop (lines 126-130)
Replace:
```c
for(idx = 0; idx < subrange; idx++)
{
    local_sum += DefaultProfile[idx];
    default_sum[idx] = local_sum;
}
```
With the same pattern already used by non-rank-0 (but with `t_start = 0`):
```c
double t_start = 0.0;
double vel = 0.0;
int table_idx = 0;
for (long i = 0; i < (long)(subrange_time / dt); i++) {
    double t = t_start + i * dt;
    vel += faccel(t) * dt;
    if ((i + 1) % steps_per_rank == 0) {
        default_sum[table_idx] = vel;
        table_idx++;
    }
}
local_sum = vel;
```

### 3. Add `memcpy` between Phase 1 and Phase 2
After the Phase 1 broadcast (line 172), copy corrected velocity into `VelProfile[]` so `fvel()` can interpolate:
```c
memcpy(VelProfile, default_sum, sizeof(double) * tablelen);
```

### 4. Convert Phase 2 non-rank-0 loop (lines 204-208)
Replace:
```c
for(idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
{
    local_sum += default_sum[idx];
    default_sum_of_sums[idx] = local_sum;
}
```
With:
```c
double t_start = my_rank * subrange_time;
double dist = 0.0;
int table_idx = my_rank * subrange;
for (long i = 0; i < (long)(subrange_time / dt); i++) {
    double t = t_start + i * dt;
    dist += fvel(t) * dt;
    if ((i + 1) % steps_per_rank == 0) {
        default_sum_of_sums[table_idx] = dist;
        table_idx++;
    }
}
local_sum = dist;
```

### 5. Convert Phase 2 rank 0 loop (lines 213-217)
Same as step 4 but with `t_start = 0.0` and `table_idx = 0`:
```c
double t_start = 0.0;
double dist = 0.0;
int table_idx = 0;
for (long i = 0; i < (long)(subrange_time / dt); i++) {
    double t = t_start + i * dt;
    dist += fvel(t) * dt;
    if ((i + 1) % steps_per_rank == 0) {
        default_sum_of_sums[table_idx] = dist;
        table_idx++;
    }
}
local_sum = dist;
```

## Why the rank 0 / non-rank-0 split exists
The split is **not** because the math differs — it's because of MPI communication. Non-rank-0 processes send results; rank 0 receives and prints. The compute loop inside each branch is the same formula, just written separately. Both need the same `faccel(t)*dt` / `fvel(t)*dt` conversion.

## Files to modify
- [mpi_train.c](hello_cluster/mpi_train.c) — all changes listed above

## Verification
1. `make` in `hello_cluster/`
2. `mpirun -np 4 ./mpi_train`
3. Compare velocity/distance output against OpenMP version at same dt
