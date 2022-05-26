# GPU 架构简介

## PTX 硬件模型
> 本节来源: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#set-of-simt-multiprocessors

PTX 的硬件模型和 CUDA 的硬件模型一致(废话).

一个 GPU 上面有很多的 SM, 每个 SM 上面有很多 SP 核心, SM 负责无上下文切换开销的管理线程, 同时实现了单指令 barrier 同步机制. 这样, 可以把问题切分到极细的地步.

SM 的 SIMT 单元以 warp 粒度对并行线程进行管理和调度. SIMT warp 内部的线程会在同一个程序地址开始, 但是之后就可以自由的分支和执行.

At every instruction issue time, the SIMT unit selects a warp that is ready to execute and issues the next instruction to the active threads of the warp. A warp executes one common instruction at a time, so full efficiency is realized when all threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp serially executes each branch path taken, disabling threads that are not on that path, and when all paths complete, the threads converge back to the same execution path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjointed code paths.

> 此条和上面冲突啊，需要验证一下；如果大家分支了岂不是非常慢

### Independent Thread Scheduling

On architectures prior to Volta, warps used a single program counter shared amongst all 32 threads in the warp together with an active mask specifying the active threads of the warp. As a result, threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data, and algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from.

Starting with the Volta architecture, Independent Thread Scheduling allows full concurrency between threads, regardless of warp. With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. A schedule optimizer determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity.

Independent Thread Scheduling can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures. In particular, any warp-synchronous code (such as synchronization-free, intra-warp reductions) should be revisited to ensure compatibility with Volta and beyond. See the section on Compute Capability 7.x in the Cuda Programming Guide for further details.

