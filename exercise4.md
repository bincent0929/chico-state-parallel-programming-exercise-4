!! I'm using an RTX 4070 Super Laptop card for my testing !!

## 1. 

### a.
- It looks like for `cuda_hello.cu` the maximum amount of threads that can print is 1024. Which is the amount of threads that can be on one block for my GPU.
  - For `cuda_hello1.cu` the max amount of threads that can print hello is 864 * 1024. Which 864 being how many blocks there are on the GPU and 1024 being how many threads there are on each block.
  - "Can you use bigger blocks and threads"
    - Well, in my case, for my GPU, I would not be able to scale any more than I have. But, if I had access to more hardware, the program would be able to scale by increasing the block size and threads used for each of those blocks up to however many the hardware could provide. Obviously if you're using multiple hosts you would need to also use MPI.

### b. 
I built the `cuda_trap1.cu` program and using nearly the maximum available values I could put for `n` and the block count and threads per block, these are my results:

```c
$ ./cuda_trap1 884000 0 3.14159265358979323846264338327950288 864 1024
The area as computed by cuda is: 2.000657e+00
The area as computed by cpu is: 2.000605e+00
Device times:  min = 5.952981e-01, max = 6.454661e-01, avg = 6.299590e-01
  Host times:  min = 3.762007e-03, max = 5.332947e-03, avg = 3.930178e-03
```

#### Comparison

It looks like the Device ended up taking long on average than the host for the calculation.

#### Correctness

They did get approxemately 2.0 when `a = 0` and `b = pi`.

#### Command Line Argument Explanation

- `n`
  - The amount of trapezoids to use
- `a`
  - The start index for the function
- `b`
  - The end index for the function
- `blk_ct`
  - The amount of blocks to be used on the Device
- `th_per_blk`
  - The amount of threads that will be run on each block

#### Program Simplification

Look at `cuda_trap1.cu` and compile it for yourself inputting the amount of trapezoids in as the first and only argument in the command line.

The output should look like this:
```c
$ ./cuda_trap1 1000000
The area as computed by cuda is: 2.000982e+00
The area as computed by cpu is: 2.001454e+00
Device times:  min = 6.419001e-01, max = 7.105761e-01, avg = 6.882100e-01
  Host times:  min = 4.268169e-03, max = 5.291939e-03, avg = 4.500465e-03
```

As you can see, even with the change in arugments required and slight refactor to enable 1 million+ trapezoid processing, the Device is still slower than a single core on the host.

