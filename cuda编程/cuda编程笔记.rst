cuda编程笔记
==========================

cuda下载：https://developer.nvidia.com/cuda-toolkit-archive

cudnn下载：https://developer.nvidia.com/rdp/cudnn-archive

不同显卡计算能力查询：https://developer.nvidia.com/cuda-gpus

不同计算能力的配置：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

代码：https://github.com/PacktPublishing/Learn-CUDA-Programming.git

CUDA核心包含三个重点抽象：线程组层次、共享存储器和栅栏同步(barrier synchronization)

PCIE带宽
-----------------------
下面是 **单个通道(lane)、单个方向** 的传输速率和理论带宽（GT/s表示每秒传输的次数，扣除编码开销后才是有效带宽）：

* PCIe 1.0：2.5 GT/s，8b/10b编码，理论带宽约250 MB/s。
* PCIe 2.0：5 GT/s，8b/10b编码，理论带宽约500 MB/s。
* PCIe 3.0：8 GT/s，128b/130b编码，理论带宽约985 MB/s（约1 GB/s）。
* PCIe 4.0：16 GT/s，128b/130b编码，理论带宽约1.97 GB/s。
* PCIe 5.0：32 GT/s，128b/130b编码，理论带宽约3.94 GB/s。

.. note::

    显卡PCIE通常连接16通道(x16)，总带宽为单通道带宽乘以16，例如PCIe 3.0 x16单向约16 GB/s，PCIe 4.0 x16单向约32 GB/s。


GPU架构
--------------------
.. figure:: /images/cuda编程/cuda0.jpg

    CPU和GPU架构

.. figure:: /images/cuda编程/cuda3.jpg

1. GPU有大量的寄存器，驻留在SM上的所有线程的上下文一直保存在寄存器中，切换线程几乎没有开销；
   CPU线程切换时则需要把寄存器等上下文保存到内存中，再恢复另一个线程的上下文
2. **CUDA Threads:** cuda线程在cuda core上执行；cuda线程上下文切换比CPU线程上下文切换代价小的多；
   每个线程必须执行相同的kernel并独立工作在不同的数据上(SIMT)
3. **CUDA blocks:** CUDA线程被分组为一个逻辑实体，称为CUDA block,CUDA block在单个流式多处理器(Streaming Multiprocessor ( **SM** ))
   上执行,一个block在一个SM上运行，也就是说，一个block内的所有线程只能在一个SM的核心上执行，而不能在其他SM的核心上执行。
   每个GPU可能有一个或多个SM，因此，为了有效地利用整个GPU，用户需要将并行计算分为block和线程。
4. **GRID/kernel:** CUDA块被分组为一个逻辑实体，称为CUDA GRID。然后在设备上执行一个CUDA GRID。

参考：https://zhuanlan.zhihu.com/p/413145211

架构发展顺序(按发布时间) | 英文：Tesla → Fermi → Kepler → Maxwell → Pascal → Volta → Turing → Ampere → Ada Lovelace → Hopper → Blackwell

架构发展顺序(按发布时间) | 中文：特斯拉 → 费米 → 开普勒 → 麦克斯韦 → 帕斯卡 → 伏特 → 图灵 → 安培 → 阿达·洛芙莱斯 → 霍珀 → 布莱克韦尔

使用GPU的方式
---------------------
.. figure:: /images/cuda编程/cuda4.jpg

.. figure:: /images/cuda编程/cuda2.jpg

    使用GPU的三种方式

1. 使用现成的库，如libtorch等
2. 使用OpenACC指令来获得快速加速结果和可移植性
3. 通过使用C、C++、Fortran、Python等语言的结构来深入研究CUDA，以获得最高的性能和灵活性


helloword
----------------------
.. literalinclude:: code/helloword.cu
    :language: cu

* ``__global__`` 函数在设备上执行，但是在主机上调用，函数返回值必须是void
* ``<<<,>>>``  告诉编译器调用的是设备端的函数，并指定调用的线程数
* ``threadIdx.x, blockIdx.x`` :线程在块内的索引和块在网格内的索引。threadIdx只在块内唯一，
  线程在整个网格中的唯一ID为 ``blockIdx.x * blockDim.x + threadIdx.x``
* ``cudaDeviceSynchronize()``: CUDA中所有的内核调用都是异步。在调用内核后，主机不会等待内核完成，而是立即开始执行下一条指令。
  这并不奇怪，因为这是一个异构环境，所以主机和设备都可以并行运行，以利用现有的处理器类型。
  如果主机需要等待设备完成，CUDA提供了API作为CUDA编程的一部分，使主机代码等待设备功能的完成。
  其中一个API是cudaDeviceSynchronize，它等待所有先前对设备的调用完成。


错误检查
------------------------------
所有的运行时函数都返回错误码，但对于异步函数，由于会在任务结束前返回，因此错误码不能报告异步调用的错误；
错误码只报告在任务执行之前的错误，典型的错误有关参数有效性；
如果异步调用出错，错误将会在后面某个无关的函数调用中出现。
唯一能够检查异步调用出错的方式是通过在异步调用函数后面使用cudaDeviceSynchronize()同步（或使用其它同步机制），
然后检查cudaDeviceSynchronize()的返回值。运行时为每个主机线程维护着一个初始化为cudaSuccess的错误变量，
每次错误发生（可以是参数不正确或异步错误）时，该变量会被错误码重写。

* ``cudaPeekAtLastError()`` 返回这个变量，
* ``cudaGetLastError()`` 会返回这个变量，并将它重新设置为cudaSuccess。

内核发射不返回任何错误码，所以应当在内核发射后立刻调用cudaGetLastError()或cudaPeekAtLastError()检测发射前错误。
为保证cudaGetLastError()返回的错误值不是由于内核发射之前的错误导致的，
必须保证运行时错误变量在内核发射前被设置为cudaSuccess，可以通过在内核发射前调用cudaGetLastError()实现。
内核发射是异步的，因此为了检测异步错误，应用必须在内核发射和cudaGetLastError()或cudaPeekAtLastError()之间同步。
注意cudaStreamQuery()和cudaEventQuery()可能返回的cudaErrorNotReady不被视为错误，
因此不会被cudaPeekAtLastError()或cudaGetLastError()报告。


cudaError_t
```````````````````````
cudaError_t有以下类型：

* cudaSuccess：函数执行成功
* cudaErrorInvalidValue：无效的参数值
* cudaErrorMemoryAllocation：内存分配失败
* cudaErrorInitializationError：初始化失败
* cudaErrorLaunchFailure：内核执行过程中在设备上发生异常（例如访问非法地址）
* cudaErrorInvalidDevicePointer：无效的设备指针
* cudaErrorInvalidMemcpyDirection：无效的内存拷贝方向
* cudaErrorUnknown：未知错误

编程模型
------------------------

内核
````````````
CUDA通过允许程序员定义称为内核的C函数扩展了C，内核调用时会被N个CUDA线程执行N次
（译者注：这句话要好好理解，其实每个线程只执行了一次），这和普通的C函数只执行一次不同。

内核使用__global__声明符定义，使用一种新 ``<<<...>>>`` 执行配置语法指定执行某一指定内核的线程数。
每个执行内核的线程拥有一个独一无二的线程ID，可以通过内置的threadIdx变量在内核中访问（译者注：这只说明在块内是唯一的，并不一定是全局唯一的）。

内核参数： ``<<<gridDim, blockDim, sharedMem, stream>>>``

* gridDim：网格中线程块的数量（int或dim3）
* blockDim：每个线程块中线程的数量（int或dim3）
* sharedMem：每个线程块动态分配的共享内存字节数，可省略，默认为0
* stream：内核所在的流，可省略，默认为默认流

.. literalinclude:: code/example1.cu
    :language: cu

.. note:: 

    cuda还有其他几个关键字：

    * **__global__** 定义内核函数：在设备上执行，从主机上调用（计算能力3.5及以上也可以通过动态并行从设备上调用），函数返回值必须是void。
    * **__device__** 关键字用于指定该函数在设备上执行，即在GPU上执行，而不在主机上执行。这意味着该函数被编译为设备代码，只能从设备代码中调用。
    * **__host__** 该关键字用于指定该函数在主机上执行，即在CPU上执行。这意味着该函数被编译为主机代码，并且只能从主机上调用。不加任何修饰的函数默认就是__host__。
    * **__host__ __device__** 两者同时使用时，函数会被同时编译为主机代码和设备代码，可以在主机和设备上分别调用。
    * **__shared__** 声明的变量分配在共享内存中，每个线程块各有一份，块内所有线程共享。共享内存位于SM片上，速度接近L1缓存。
    * **__constant__** 声明的变量分配在常量内存中，必须定义在文件作用域（不能在函数内部）。设备端只读，主机端通过cudaMemcpyToSymbol()写入。
    * **__managed__** 声明的变量分配在统一内存中，必须定义在文件作用域，主机和设备都可以直接访问。


context
```````````````````
CUDA 上下文类似于CPU的进程。所有资源和在驱动程序API 中执行的操作都封装在CUDA 上下文内，在销毁上下文时，系统将自动清理这些资源。
除了模块和纹理参考之类的对象外，每个上下文都有自己不同的地址空间。因而，不同上下文的CUdeviceptr 值将引用不同的存储器空间。

一个主机线程在某时只能有一个当前设备上下文。当使用cuCtxCreate()创建上下文时，它将成为主机调用线程的当前上下文。
如果有效上下文不是线程的当前上下文，在该线程中操作的CUDA 函数（不涉及设备模拟或上下文管理的大多数函数）将返回CUDA_ERROR_INVALID_CONTEXT。

每个主机线程都有一个当前上下文堆栈。cuCtxCreate() 将新上下文压入栈顶。可调用cuCtxPopCurrent() 分离上下文与主机线程。
随后此上下文将成为“游魂（floating）”上下文，可作为任意主机线程的当前上下文入栈。cuCtxPopCurrent() 还可重建之前的当前上下文（如果有）。

.. note::

    下面介绍的cuCtxAttach()/cuCtxDetach()引用计数机制来自早期的CUDA版本，这两个函数已被废弃。
    现在使用运行时API时，一般直接使用每个设备的主上下文(primary context)，驱动API中对应的接口是cuDevicePrimaryCtxRetain()/cuDevicePrimaryCtxRelease()。

此外还会为每个上下文维护一个引用计数（usage count）。cuCtxCreate()创建一个将引用计数为1的上下文。
cuCtxAttach()递增计数，而cuCtxDetach() 则递减。当调用cuCtxDetach()时计数为0 或cuCtxDestroy()，上下文将被销毁。

引用计数有利于同一上下文中第三方授权代码间的互操作。比如，如果载入了三个使用相同上下文的库，则每个库都将调用cuCtxAttach() 来递增计数，
并在库不再使用该上下文时调用cuCtxDetach() 递减计数。对大多数库来说，应用应当在载入或初始化库之前创建一个上下文，通过这种方式，
应用可使用自己的启发式（heuristics）方法来创建上下文，库只需在传递给它的上下文中简单操作。
希望创建自己的上下文的库（其客户端并不了解这种情况，并且可能已经创建或未创建自己的上下文）可使用cuCtxPushCurrent()和cuCtxPopCurrent()

线程层次
```````````````````````
cuda线程层次分为 **线程** ， **束(warp)**, **块(block)**， **栅格(grid)**

block是多个线程的集合，grid是多个block的集合

.. image:: /images/cuda编程/cudathread1.jpg

当我们启动一个CUDA内核时，一个或多个CUDA thread block会在GPU的流式多处理器(SM)上并行执行。
一个流式多处理器可以根据资源的可用性，运行多个block，这取决于资源的可用性。一个线程块中的线程数量是变化，grid中的block数也是变化。
流式多处理器根据GPU资源，以不确定的顺序并行地执行线程块。

.. image:: /images/cuda编程/cudathread2.jpg

CUDA流式多处理器(SM)将CUDA线程以 **32个为一组** 进行控制。一个组被称为一个 **束(warp)** 。
通过这种方式，一个或多个 **束(warp)** 配置一个block。

.. image:: /images/cuda编程/cudathread3.jpg

.. note:: 

    warp中的32个线程同时执行同一条指令（锁步执行），不同warp之间的执行顺序是不确定的，
    因此block中不同warp的线程之间没有执行顺序的保证，需要用__syncthreads()同步

.. note:: 

    cuda占用率是指活动的CUDA warps与每个流式多处理器可以并发执行的最大warps的比率

线程
~~~~~~~~~~~~~ 

cuda中线程通过内置变量threadIdx进行索引，threadIdx是一个有3个分量(threadIdx.x,threadIdx.y,threadIdx.z)的向量，
所以线程可以使用一维，二维，三维索引标识

线程索引和线程ID直接相关：对于一维的块，它们相同；对于二维长度为(Dx,Dy)的块，
线程索引为(x,y)的线程ID是(x+yDx)；对于三维长度为(Dx,Dy,Dz)的块，索引为(x,y,z)的线程ID为(x+yDx+zDxDy)
（这和我们使用C数组的方式不一样，大家注意理解）

.. image:: /images/cuda编程/cuda1.jpg

.. literalinclude:: code/example2.cu
    :language: cu

束(warp)
~~~~~~~~~~~~~~~~~~~~~~~~~~
一个warp通常由32个线程组成，并且在一个时钟周期内执行相同的指令，即"Single Instruction"。
这32个线程被划分为不同的lane（或称为"lanes"），每个lane负责处理指令中的一部分数据或执行特定的计算。
具体而言，一个warp内的所有线程会并行地执行相同的指令流，但它们所处理的数据可能不同。
lane_idx是每个warp中的线程对应的编号，范围为[0,31]。

.. code-block:: cpp

    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x & (warpSize - 1);

.. note:: 

    lane_idx的计算方式只适用于一维线程块；多维线程块需要先计算线性化的线程ID：
    ``threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y``

**branch divergence**:如果 warp 执行时遇到条件语句或分支语句，其线程可以分流并序列化，以执行每个条件，
这被称为分支分歧(branch divergence),会严重影响性能。

.. image:: /images/cuda编程/cudathread5.jpg

解决分支分歧的方法有：

1. 使用不同的warp来处理不同的分支
2. 凝聚分支部分，减少warp中的分支
3. 缩短分支部分；只对关键部分进行分支
4. 重新排列数据（即移位、合并等）
5. 在合作小组中使用 tiled_partition 对小组进行分区


warp原语编程
''''''''''''''''''''''''''''''''
https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

CUDA 9.0引入了新的warp同步编程。这一重大变化旨在避免CUDA编程依赖隐式warp同步操作和显式处理同步目标。这有助于防止在warp同步操作中出现不注意的竞争条件和死锁

历史版本，CUDA只为线程块中的CUDA线程提供了一个显式同步API( ``__syncthreads()`` )，它依赖于warp的隐式同步。下图显示了CUDA线程块操作的两个级别的同步:

.. image:: /images/cuda编程/cuda_wrap1.png

然而，从Volta架构开始，GPU有一个增强的线程控制模型，其中每个线程可以执行不同的指令，同时保持其SIMT编程模型。下图显示了它是如何变化的:

.. image:: /images/cuda编程/cuda_wrap2.png

在Pascal及之前的架构(左)中，线程是在warp级别调度的，它们在warp级别内隐式同步。因此，CUDA线程在warp中隐式同步。然而，这可能会导致意外的死锁。

Volta架构改变了这一点，引入了独立的线程调度。这种控制模型使每个CUDA线程都有自己的程序计数器，warp内的线程可以在不同的分支上交错执行，
不再保证同一warp内的线程始终同步。
在这个模型中，我们必须使用显式的同步API来指定哪些线程参与warp内的同步和数据交换。

CUDA 9引入了显式的warp级原语函数:

* 识别活动线程： ``__activemask()``

.. code-block:: cpp

    __global__ void activeMaskExample() {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 返回值的第i位表示warp中lane i的线程当前是否处于活跃状态
        unsigned int active_mask = __activemask();
        int lane_idx = threadIdx.x & (warpSize - 1);  // 必须用lane编号，threadIdx.x可能大于31
        if (active_mask & (1u << lane_idx)) {
            printf("Thread %d is active.\n", tid);
        }
    }

* 同步数据交换（投票和匹配）： ``__all_sync()`` ``__any_sync()`` ``__uni_sync()`` ``__ballot_sync()`` ``__match_any_sync()``  ``__match_all_sync()``
* 同步数据交换（shuffle）: ``__shfl_sync()``  ``__shfl_up_sync()`` ``__shfl_down_sync()`` ``__shfl_xor_sync()``
* 线程同步: ``__syncwarp()``

这些函数的第一个参数都是mask，指定参与操作的线程，例如 ``0xffffffff`` 表示warp中的全部32个线程。


块(block)
~~~~~~~~~~~~~ 

由于块内的所有线程必须存在于同一个处理器核心(SM)中且共享该核心有限的存储器资源，因此，一个块内的线程数目是有限的，
目前的GPU上每个块最多1024个线程。
然而，一个内核可被多个同样大小的线程块执行，所以总的线程数等于每个块内的线程数乘以线程块数。

多个块之间的线程使用内置变量blockIdx和blockDim进行索引。

``index = threadIdx.x + blockIdx.x * blockDim.x;``

.. image:: /images/cuda编程/cudathread4.jpg


块内线程可通过共享存储器和同步执行协作，共享存储器可以共享数据，同步执行可以协调存储器访问。
更精确一点说，可以在内核中调用 ``__syncthreads()`` 内置函数指明 **同步点** ； ``__syncthreads()`` 起栅栏的作用，
在其调用点，块内线程必须等待，直到所有线程都到达此点才能向前执行。

.. note:: 

    block内的不同warp是乱序执行的

网格(Grid)
~~~~~~~~~~~~~ 

线程 **块(block)** 被组织成一维、二维或三维的线程 **网格（Grid）** ，
如上图所示。一个网格内的线程块数往往由被处理的数据量而不是系统的处理器数决定，前者往往远超后者。

线程块内线程数和网格内线程块数由 ``<<<block, thread, shared_memory, stream>>>`` 语法确定，参数可以是整形或者dim3类型。

网格内的每个块可以通过一维、二维或三维索引唯一确定，在内核中此索引可通过内置的blockIdx变量访问。
块的尺寸(dimension)可以在内核中通过内置变量blockDim访问。    

为了处理多个块，扩展前面的MatAdd()例子后，代码成了下面的样子.

.. literalinclude:: code/example3.cu
    :language: cu

一个长度为16*16（256线程）的块，虽然大小是任意选择的，但是很常见。像以前一样，创建了内有足够的块的网格，使得一个线程处理一个矩阵元素。
为简便起见，此例假设网格每一维上的线程数可被块内对应维上的线程数整除，尽管这并不是必需。

线程块必须独立执行：而且能够以任意顺序，串行或者并行执行。这种独立性要求使得线程块可以以任何顺序在任意数目核心上调度。

.. note:: 

    grid中block是乱序执行的

相关API
~~~~~~~~~~~~~ 

cudaOccupancyMaxActiveBlocksPerMultiprocessor()用于查询给定内核和线程块大小时，每个SM（Streaming Multiprocessor）
上可以同时驻留的最大线程块数，以便在使用CUDA内核时选择合适的启动配置。

该函数返回cudaError_t表示执行结果，查询结果写入numBlocks指向的变量。

.. code-block:: cpp

    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        int *numBlocks,
        const void *func,
        int blockSize,
        size_t dynamicSMemSize
    );
    // 需要指定额外选项时使用带flags的版本
    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        int *numBlocks,
        const void *func,
        int blockSize,
        size_t dynamicSMemSize,
        unsigned int flags
    );

* numBlocks：输出参数，每个SM上可同时驻留的最大线程块数。
* func：内核函数。
* blockSize：线程块的大小（每个块的线程数）。
* dynamicSMemSize：每个线程块动态共享内存的大小（字节）。
* flags：查询选项，例如cudaOccupancyDefault、cudaOccupancyDisableCachingOverride。

用 numBlocks * blockSize / 每个SM最大线程数，即可得到该配置下的理论占用率。
另外，cudaOccupancyMaxPotentialBlockSize()可以直接给出使占用率最大的线程块大小。


cudaDeviceGetAttribute()函数是CUDA Toolkit中用于获取单个设备属性的函数，比cudaGetDeviceProperties()一次性获取所有属性要快。

该函数返回cudaError_t表示执行结果。如果函数执行成功，则返回cudaSuccess，否则返回一个错误代码。

.. code-block:: cpp

    cudaError_t cudaDeviceGetAttribute(
        int *value,
        cudaDeviceAttr attr,
        int device
    );

* value：指向一个整数类型的指针，用于存储设备属性的值。
* attr：要获取的设备属性。
* device：要获取属性的设备编号。

cudaDeviceGetAttribute()函数可以用于获取设备的各种属性，例如Warp大小、每个块的最大线程数、SM数量、共享内存大小等。
这些属性可以用于优化CUDA程序的性能和资源使用。显存总量不属于这里的属性，可以通过cudaMemGetInfo()或cudaDeviceProp::totalGlobalMem获取。

以下是一些常用的设备属性（完整列表见CUDA Runtime API文档中的cudaDeviceAttr）：

* cudaDevAttrMaxThreadsPerBlock：每个线程块的最大线程数。
* cudaDevAttrMaxBlockDimX / cudaDevAttrMaxBlockDimY / cudaDevAttrMaxBlockDimZ：线程块在X/Y/Z维度上的最大尺寸。
* cudaDevAttrMaxGridDimX / cudaDevAttrMaxGridDimY / cudaDevAttrMaxGridDimZ：网格在X/Y/Z维度上的最大尺寸。
* cudaDevAttrMaxSharedMemoryPerBlock：每个线程块可用的最大共享内存（字节）。
* cudaDevAttrMaxSharedMemoryPerBlockOptin：通过cudaFuncSetAttribute()显式申请后，每个线程块可用的最大共享内存（字节）。
* cudaDevAttrMaxSharedMemoryPerMultiprocessor：每个SM的最大共享内存（字节）。
* cudaDevAttrTotalConstantMemory：常量内存大小（字节）。
* cudaDevAttrWarpSize：warp大小（线程数）。
* cudaDevAttrMaxPitch：内存拷贝允许的最大pitch（字节）。
* cudaDevAttrMaxRegistersPerBlock：每个线程块可用的最大32位寄存器数。
* cudaDevAttrMaxRegistersPerMultiprocessor：每个SM的最大32位寄存器数。
* cudaDevAttrMaxThreadsPerMultiProcessor：每个SM最多驻留的线程数。
* cudaDevAttrClockRate：峰值时钟频率（kHz）。
* cudaDevAttrMemoryClockRate：峰值显存时钟频率（kHz）。
* cudaDevAttrGlobalMemoryBusWidth：显存总线位宽（bit）。
* cudaDevAttrL2CacheSize：L2缓存大小（字节）。
* cudaDevAttrMultiProcessorCount：SM的数量。
* cudaDevAttrComputeCapabilityMajor / cudaDevAttrComputeCapabilityMinor：计算能力的主/次版本号。
* cudaDevAttrKernelExecTimeout：内核执行是否有运行时间限制（例如连接显示器的显卡）。
* cudaDevAttrIntegrated：是否为集成显卡。
* cudaDevAttrCanMapHostMemory：是否支持将主机内存映射到设备地址空间。
* cudaDevAttrComputeMode：设备的计算模式。
* cudaDevAttrConcurrentKernels：是否支持多个内核并发执行。
* cudaDevAttrAsyncEngineCount：异步拷贝引擎的数量（取代已废弃的cudaDevAttrGpuOverlap）。
* cudaDevAttrUnifiedAddressing：是否支持统一虚拟地址空间(UVA)。
* cudaDevAttrManagedMemory：是否支持统一内存(managed memory)。
* cudaDevAttrConcurrentManagedAccess：CPU和GPU是否可以并发访问统一内存。
* cudaDevAttrTextureAlignment：纹理基地址的对齐要求（字节）。
* cudaDevAttrMaxTexture1DWidth：一维纹理的最大宽度。
* cudaDevAttrMaxTexture2DWidth / cudaDevAttrMaxTexture2DHeight：二维纹理的最大宽度/高度。
* cudaDevAttrMaxTexture3DWidth / cudaDevAttrMaxTexture3DHeight / cudaDevAttrMaxTexture3DDepth：三维纹理的最大宽度/高度/深度。
* cudaDevAttrMaxSurface2DWidth / cudaDevAttrMaxSurface2DHeight：二维表面(surface)的最大宽度/高度。


内存层次
```````````````````````````
.. figure:: /images/cuda编程/cudamem1.jpg

.. figure:: /images/cuda编程/cudamem2.jpg


GPU内存分类：

* 全局内存/设备内存
* 共享内存(Shared memory)
* 只读数据/缓存
* 寄存器
* Pinned memory(锁定内存)
* Unified memory(统一内存)

全局内存/设备内存
~~~~~~~~~~~~~~~~~~~~~~~~~~~
全局内存/设备内存对内核中所有的线程都是可见的。CPU不能直接解引用设备内存指针，但可以通过cudaMemcpy等API读写它。

程序员用cudaMalloc和cudaFree明确地管理分配和删除全局内存。
全局内存是所有使用cudaMemcpy API从CPU传输的内存的默认暂存区域。

共享内存
~~~~~~~~~~~~~~~
共享内存在CUDA的内存层次结构中一直扮演着重要的角色，被称为用户管理缓存。
这为用户提供了一种机制，使他们可以以聚集的方式从全局内存中读/写数据，并将其存储在共享内存中，这就像一个缓存，但可以由用户控制

共享内存只对同一个block中的线程可见。一个block中的所有线程都能看到共享变量的相同版本。

共享内存具有与CPU缓存类似的优点；然而，CPU缓存不能被显式管理，而共享内存可以

和全局内存相比，共享内存有更高的带宽和更低的延迟

CUDA程序员可以使用共享变量来保存数据，这些数据在内核的执行阶段被多次重复使用。

共享内存的一个应用实例为矩阵转置

.. literalinclude:: code/matrix_transpose.cu
    :language: cu


**bank conflict(分组冲突)**:

bank(分组)：共享内存被分组管理，以实现更高的带宽。Volta GPU 有 32 个bank，每个bank 4 字节宽

一个warp 中的多个线程同时访问一个bank会导致bank conflict。换句话说，当一个warp中的两个或两个以上的线程访问同一个bank中的不同4字节字时，就会发生bank冲突。
发生冲突时，对同一bank的访问会被串行化。如果多个线程访问的是同一个4字节字，则会广播给这些线程，不会产生bank冲突。

上面的矩阵转置例子中，共享内存声明为 ``[BLOCK_SIZE][BLOCK_SIZE + 1]`` ，多出的一列就是为了让同一列的元素错开到不同的bank，避免按列访问时的bank冲突。


.. figure:: /images/cuda编程/cudasharedmemory.jpg

    共享内存逻辑视图

.. image:: /images/cuda编程/cudasharedmemory1.jpg

.. image:: /images/cuda编程/cudasharedmemory2.jpg

.. image:: /images/cuda编程/cudasharedmemory3.jpg


只读数据/缓存
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Constant memory
* Texture memory

只读缓存用来存储在内核执行期间只读的数据，早期需要通过纹理内存(texture memory)来使用，用户可以明确地调用纹理API来使用只读缓存。

只读数据对GPU中网格(Grid)中的所有线程都是可见的。

在内核执行期间设备端只能读取这些数据，CPU可以在内核启动前写入（例如通过cudaMemcpyToSymbol()写常量内存，或把数据拷贝到绑定纹理的内存中）。

在较新的GPU架构上，开发者可以利用只读缓存，而无需明确使用CUDA纹理API。从计算能力3.5开始，
如果内核的指针参数标记为 const __restrict__ ，编译器就可以把对它的读取通过只读缓存的数据路径加载。
开发者还可以通过 __ldg() 内置函数显式地通过只读缓存加载数据。

缓存的一个应用是对图片的缩放

.. literalinclude:: code/image_scaling.cu
    :language: cu

寄存器
~~~~~~~~~~~~~~~~~~~
CPU和GPU架构之间的根本区别之一是与CPU相比，GPU有大量的寄存器。这有助于线程将大部分的数据保存在寄存器中，因此减少了上下文切换的延时。

寄存器的作用范围是单个线程，每个线程都可以访问其变量的私有副本，而其他线程的私有变量不能被访问。

作为内核的一部分被声明的局部变量被存储在寄存器中。临时变量也被存储在寄存器中。

每个SM中寄存器个数是固定的。在编译过程中，编译器（nvcc）试图找到每个线程的最佳寄存器数量。
如果寄存器的数量不足(这通常发生在CUDA内核很大，有很多局部变量和中间计算的情况下),
数据会被推送到本地内存中，本地内存可能位于L1/L2高速缓存中，甚至是内存层次中更低的位置，比如全局内存。这也被称为寄存器溢出。
每个线程的寄存器数量在一个SM上有多少块和线程可以被激活方面起着重要作用。

如果寄存器限制了SM上可以调度的线程数量，那么开发者应该考虑重组代码，将内核拆分成两个或更多，
或者用nvcc的 ``--maxrregcount`` 选项或 ``__launch_bounds__()`` 限制每个线程使用的寄存器数量（可能导致寄存器溢出）

.. image:: /images/cuda编程/cuda_register.jpg

Pinned memory(锁定内存)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用malloc申请的内存是可分页内存，这意味着，如果需要的话，被映射为页的内存可以被其他应用程序或操作系统本身换出。

默认情况下，GPU不会访问可分页内存。因此，当调用内存的转移时，CUDA驱动会分配临时的Pinned memory，
将数据从默认的可分页内存复制到这个临时的Pinned memory，然后通过直接内存访问（DMA, Direct Memory Access）将其传输到设备上。
直接使用Pinned memory可以省去这次额外的拷贝，提高传输带宽，并且是cudaMemcpyAsync()与内核执行重叠的前提。

使用cudaMallocHost申请Pinned memory。

Unified memory(统一内存)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
UM为用户提供了一个单一内存空间的视图，该空间可以被系统中的所有GPU和CPU访问。

就像全局内存访问，如果以非协同的方式进行，会导致糟糕的性能。UM功能如果使用方式不当，也会导致应用程序整体性能下降。

使用cudaMallocManaged()申请UM

从Pascal卡开始，cudaMallocManaged()并不分配物理内存，而是基于首次使用的方式来分配内存。
如果GPU第一次使用到这个变量，那么这个内存页面就会被分配并映射到GPU的页面表中。
否则，如果CPU首先接触到该变量，它将被分配并映射到CPU。

.. image:: /images/cuda编程/cuda_unified_memory.jpg


``cudaMemPrefetchAsync`` 是CUDA 编程中用于实现统一内存异步预取的函数。
该函数把统一内存中的数据提前迁移到指定的处理器（某个GPU，或者CPU）上，避免访问时才触发缺页迁移，以优化数据访问性能。
异步内存预取允许 CPU 和 GPU 并行执行，从而最大程度地利用硬件资源。

.. code-block:: cpp

    cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, 
                                    int dstDevice, cudaStream_t stream = 0);
    // 注：较新的CUDA版本新增了以cudaMemLocation指定目标位置的接口，签名可能与此不同，以所用版本的官方文档为准
    // devPtr：由cudaMallocManaged()等分配的统一内存的地址，指定要预取的数据的起始地址。
    // count：预取的数据量，以字节为单位。可以是一个或多个数据元素的大小，具体取决于应用程序需求。
    // dstDevice：目标设备的 ID，指定数据预取的目标设备。如果在单个设备上运行，通常将其设置为 0。
    //            预取到CPU时使用cudaCpuDeviceId（值为-1）
    // stream（可选）：指定一个 CUDA 流，用于执行预取操作。
    //                默认值为 0，表示使用默认的 CUDA 流（也称为 NULL 流）。
    //                你可以通过使用不同的 CUDA 流来实现并行的数据预取和计算。


``cudaMemAdvise`` 是 CUDA Runtime API 中的函数之一，它用于向 CUDA 运行时库提供统一内存的使用建议，以优化统一内存的放置和迁移。
该函数允许开发人员为内存操作提供有关数据访问模式的提示，以提高性能和效率。

需要注意的是， ``cudaMemAdvise`` 函数提供了一种优化内存使用的手段，但并不保证在所有情况下都能带来性能提升。
正确使用内存建议需要结合实际情况和性能测试，以确定最佳的内存管理策略。


.. code-block:: cpp

    cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device);
    // devPtr：统一内存的地址，指定要提供建议的内存起始地址。
    // count：内存区域的大小，以字节为单位。
    // advice：一个枚举类型，指定要提供的内存使用建议。可选值如下：
    // cudaMemAdviseSetReadMostly：数据主要用于读取。读取它的处理器会各自保存一份只读副本，写入时所有副本失效。
    // cudaMemAdviseUnsetReadMostly：取消之前的 "cudaMemAdviseSetReadMostly" 建议。
    // cudaMemAdviseSetPreferredLocation：把数据的首选物理位置设置为device，尽量不把数据从这里迁移走。
    // cudaMemAdviseUnsetPreferredLocation：取消之前的 "cudaMemAdviseSetPreferredLocation" 建议。
    // cudaMemAdviseSetAccessedBy：数据会被device访问，让device的页表始终保持对这块数据的映射，避免访问时缺页。
    // cudaMemAdviseUnsetAccessedBy：取消之前的 "cudaMemAdviseSetAccessedBy" 建议。
    // device：建议所针对的设备 ID，CPU使用cudaCpuDeviceId。


异构编程
```````````````````````````
CUDA编程模型假设CUDA线程在物理上独立的设备上执行，设备作为主机的协处理器，主机运行C程序。
例如，内核在GPU上执行，而C程序的其它部分在CPU上执行就是这种模式。

CUDA编程模型同时假设主机和设备各自都维护着自己独立的DRAM存储器空间，各自被称为主机存储器空间和设备存储器空间。
因此，程序通过调用CUDA 运行时，来管理对内核可见的全局、常量和纹理存储器空间。
这包括设备存储器分配和释放，也包括在主机和设备间的数据传输。

计算能力
```````````````````
设备的计算能力由主修订号和次修订号定义。

主修订号相同的设备基于相同的核心架构：

* 1.x：Tesla
* 2.x：Fermi
* 3.x：Kepler
* 5.x：Maxwell
* 6.x：Pascal
* 7.0/7.2：Volta；7.5：Turing（基于Volta的增量更新）
* 8.0/8.6/8.7：Ampere；8.9：Ada Lovelace（基于Ampere的增量更新）
* 9.0：Hopper
* 10.x/12.x：Blackwell

次修订号对应着对核心架构的增量提升，也可能包含了新特性。

支持CUDA的GPU列出了所有支持CUDA的设备和它们的计算能力。

nvcc编译参数：

* --resource-usage：编译时显示GPU资源分配（每个内核使用的寄存器、共享内存等）
* --maxrregcount=24：限制每个线程最多使用24个寄存器

编程接口
----------------
* CUDA上下文－类似设备上的主机进程
* CUDA模块－类似设备上的动态链接库

PTX
```````````````````
CUDA PTX（Parallel Thread Execution）是一种中间表示（IR），用于在NVIDIA GPU上执行并行计算。
它是一种高级汇编语言，可以在编译时将CUDA C/C++代码转换为PTX代码，然后在运行时将PTX代码转换为GPU架构特定的机器代码。
PTX代码是可移植的，可以在不同的GPU架构上运行，因此它可以提高代码的可移植性和可重用性。
此外，PTX代码还可以进行手动优化，以提高性能和效率。

nvcc编译
`````````````````````
内核可以使用PTX编写，PTX就是CUDA指令集架构。但通常使用像C++一样的高级语言编写内核更有效。
无论是使用PTX还是高级语言，内核都必须使用nvcc编译成二进制代码才能在设备上执行。

nvcc是一个编译器驱动，它简化了C或PTX的编译流程：它提供了简单熟悉的命令行选项，
同时通过调用一系列实现了不同编译步骤的工具集来执行它们。

编译流程
~~~~~~~~~~~~~~~~~~~
nvcc可编译同时包含主机代码（在主机上执行的代码）和设备代码（在设备上执行的代码）的源文件。
nvcc的基本流程包括分离主机和设备代码然后：

* 将设备代码编译成汇编形式（PTX代码）或者二进制形式（cubin对象）
* 将执行配置节引入的<<<;>>>语法转化为必要的CUDA C运行时函数调用以加载和启动每个已编译的内核（来自PTX代码或者cubin对象）

**即时编译：** 任何在运行时被应用加载的PTX代码会被设备驱动进一步编译成二进制代码，这称为即时编译

即时编译增加了应用加载时间，但允许应用从最新编译器改进中获益，也是应用能够在未来硬件上运行的唯一方法，这些硬件在应用编译时还不存在。

当设备驱动为某些应用即时编译某些PTX代码，它自动缓存生成的二进制代码的一个副本以避免在以后调用应用时重复编译。
当设备驱动升级后该缓存（称为计算缓存）自动失效，所以应用能够从设备驱动内置的新的即时编译器获益。

环境变量可用于控制即时编译：

* 设置CUDA_CACHE_DISABLE为1使缓存失效（也就是没有二进制代码增加到缓存或从缓存中检索）
* CUDA_CACHE_MAXSIZE以字节为单位指定了计算缓存的大小（默认大小随驱动版本变化，以官方文档为准，最大为4 GB）；大小超过缓存尺寸的二进制代码不会被缓存；需要时会清理旧的二进制代码以为新二进制代码腾出空间。
* CUDA_CACHE_PATH指定了计算缓存文件存储的目录，Linux系统上默认为~/.nv/ComputeCache
* 设置CUDA_FORCE_PTX_JIT为1强制设备驱动忽略任何嵌入在应用中的二进制代码而即时编译嵌入的PTX代码；
  如果内核没有嵌入的PTX代码，加载失败；这个环境变量可以用于验证应用中是否嵌入了PTX代码和即时编译是否如预期工作以保证应用能够和将来的设备向前兼容。

.. note::

    早期的nvcc可以用-m32/-m64选择以32位或64位模式编译设备代码，CUDA 12已不再支持32位编译。

CUDA C运行时
`````````````````````
cudart动态库是运行时的实现，它包含在应用的安装包里，所有的函数前缀都是cuda。

初始化
~~~~~~~~~~~~~~~~~~~
运行时在初次调用运行时函数时隐式初始化。
在计算运行时函数调用的时间和解析初次调用运行时产生的错误码时必须牢记这点。

.. note::

    CUDA 12.0起新增了显式初始化函数cudaInitDevice()，并且cudaSetDevice()会立即初始化所选设备的主上下文。
    因此可以在程序开始时调用cudaSetDevice()或cudaFree(0)提前完成初始化，避免把初始化时间计入第一次CUDA调用。

在初始化时，运行时为系统中的每个设备建立一个上下文。这个上下文作为设备的主要上下文，被应用中的主机线程共享。
这些都是隐式发生的，运行时并没有将主要上下文展示给应用。

当主机线程调用cudaDeviceReset()时，这销毁了主机线程操作的设备的主要上下文。
任何以这个设备为当前设备的主机线程调用的运行时函数将为设备重新建立一个主要上下文。

设备存储器
~~~~~~~~~~~~~~~
设备存储器即显存。

设备存储器可被分配为线性存储器或CUDA数组。CUDA数组是不透明的存储器布局，为纹理获取做了优化。

线性存储器存在于设备的统一地址空间中（早期计算能力1.x的设备为32位地址空间，2.0为40位，现代GPU为49位或更大），
所以独立分配的存储器实体能够通过指针相互引用

线性存储器使用cudaMalloc()分配， 通过cudaFree()释放， 使用cudaMemcpy()在设备和主机间传输。

线性存储器也可以通过cudaMallocPitch()和cudaMalloc3D()分配。在分配2D或3D数组的时候，推荐使用，
因为这些分配增加了合适的填充以满足对齐要求，在按行访问时或者在二维数组和设备存储器的其它区域间复制（用cudaMemcpy2D()和cudaMemcpy3D()函数）时，
保证了最佳性能。返回的步长（pitch)必须用于访问数组元素。

.. code-block:: cu

    // 用于在设备端分配内存。
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    // devPtr：指向设备端内存的指针。
    // size：要分配的内存大小（以字节为单位）。

    // 用于在设备端和主机端分配可共享的内存
    cudaError_t cudaMallocManaged(void **devPtr, size_t size, 
                                unsigned int flags=cudaMemAttachGlobal);
    // devPtr：指向设备端和主机端共享内存的指针。
    // size：要分配的内存大小（以字节为单位）。
    // flags：内存附加标志，指定内存如何附加到设备和主机上。

    // 用于在主机端分配分页锁定(pinned)内存。
    cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
    // pHost：指向主机端内存的指针。
    // size：要分配的内存大小（以字节为单位）。
    // flags：cudaHostAllocDefault、cudaHostAllocPortable、cudaHostAllocMapped、cudaHostAllocWriteCombined，可以按位或组合。

    // 用于在设备端分配二维数组内存
    cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
    // devPtr：指向设备端内存的指针。
    // pitch：输出参数，每一行实际占用的字节数（包含对齐填充）。
    // width：每一行的宽度，以字节为单位。
    // height：行数。

    // 用于在设备端分配三维数组内存。
    cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);
    // pitchedDevPtr：指向设备端内存的指针和每一行字节数的结构体。
    // extent：数组的大小和维度。

    // 用于在设备端分配三维数组纹理内存
    cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, 
                cudaExtent extent, unsigned int flags = 0);
    // array：指向设备端数组的指针。
    // desc：数组的通道格式描述符。
    // extent：数组的大小和维度。
    // flags：数组分配标志，指定数组如何分配。

.. code-block:: cu

    // 释放由 CUDA 分配的设备端内存
    cudaError_t cudaFree(void *devPtr);
    // 如果要释放的内存是由cudaMallocManaged() 函数分配的，
    // 则需要使用cudaFree() 函数释放设备端和主机端共享的内存。
    // 如果释放成功，cudaFree()函数将返回cudaSuccess。如果释放失败，将返回相应的错误码。

.. code-block:: cu

    // 主机端和设备端之间复制数据
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    // dst 是目标内存地址，
    // src 是源内存地址，
    // count  是要复制的数据大小（以字节为单位），
    // kind是数据传输方向，可以是以下值之一：
    // cudaMemcpyHostToHost：主机端内存到主机端内存。
    // cudaMemcpyHostToDevice：主机端内存到设备端内存。
    // cudaMemcpyDeviceToHost：设备端内存到主机端内存。
    // cudaMemcpyDeviceToDevice：设备端内存到设备端内存。
    // cudaMemcpyDefault：根据指针自动推断传输方向，需要支持统一虚拟地址空间(UVA)。
    // 如果复制成功，cudaMemcpy() 函数将返回 cudaSuccess。如果复制失败，将返回相应的错误码。

    // 主机端和设备端之间复制二维数组数据
    cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, 
                size_t width, size_t height, cudaMemcpyKind kind);
    // dst 是目标内存地址，
    // dpitch 是目标内存每一行的字节数，
    // src 是源内存地址，
    // spitch 是源内存每一行的字节数，
    // width 是要复制的数据宽度（以字节为单位），
    // height 是要复制的数据高度（以行数为单位），
    // kind 是数据传输方向，同上

    // 用于在主机端和设备端之间复制三维数组数据
    cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p);
    // p 是一个指向 cudaMemcpy3DParms 结构体的指针，该结构体包含以下参数：
    // struct cudaArray *srcArray：源内存的 CUDA 数组。
    // struct cudaMemcpy3DParms::cudaPos srcPos：源内存的起始位置。
    // struct cudaPitchedPtr dstPtr：目标内存地址和每一行字节数的结构体。
    // struct cudaMemcpy3DParms::cudaExtent extent：内存的大小和维度。
    // enum cudaMemcpyKind kind：数据传输方向，可以是四种值之一,同上

    // 将数据复制到设备端的符号（__device__或__constant__变量），常用于写常量内存
    cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset = 0, 
                        cudaMemcpyKind kind = cudaMemcpyHostToDevice);
    // symbol  是要复制到的设备端变量的符号，
    // src  是源内存地址，
    // count 是要复制的数据大小（以字节为单位），
    // offset 是符号地址的偏移量（以字节为单位），
    // kind 是数据传输方向，可以是以下两种值之一：
    // cudaMemcpyHostToDevice：主机端内存到设备端内存。
    // cudaMemcpyDeviceToDevice：设备端内存到设备端内存。

    // 将数据从设备端的符号（__device__或__constant__变量）复制出来。
    cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset = 0, 
                    cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
    // dst 是目标内存地址，
    // symbol 是要复制的设备端变量的符号，
    // count 是要复制的数据大小（以字节为单位），
    // offset 是符号地址的偏移量（以字节为单位），
    // kind 是数据传输方向，可以是以下两种值之一：
    // cudaMemcpyDeviceToHost：设备端内存到主机端内存。
    // cudaMemcpyDeviceToDevice：设备端内存到设备端内存。

    // 异步复制数据，立即返回，复制操作在stream中按顺序执行。主机内存需要是分页锁定内存才能真正异步。
    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);


.. code-block:: cu

    // 获取设备端常量内存的符号地址
    cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
    // devPtr 是指向设备端内存的指针，
    // symbol 是要获取地址的设备端常量内存的符号地址。

    // 用于获取设备端常量内存的大小
    cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
    // size  是指向存储设备端常量内存大小的变量的指针，
    // symbol 是要获取大小的设备端常量内存的符号地址。


下面的代码分配了一个尺寸为width*height的二维浮点数组，同时演示了怎样在设备代码中遍历数组元素。

.. literalinclude:: code/example4.cu
    :language: cu

下面的代码分配了一个尺寸为width*height*depth的三维浮点数组，同时演示了怎样在设备代码中遍历数组元素。

.. literalinclude:: code/example5.cu
    :language: cu


共享存储器
~~~~~~~~~~~~~~~~~~
共享存储器使用__shared__限定词分配

共享存储器应当比全局存储器更快。任何用访问共享存储器取代访问全局存储器的机会应当被发掘


分页锁定主机存储器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
分页锁定主机存储器（Pinned Host Memory）是一种特殊的主机端内存，它可以通过调用cudaHostAlloc()函数来分配。
与普通的主机端内存不同，分页锁定主机存储器在分配时会被锁定在物理内存中，
这意味着它不会被操作系统交换到磁盘上，从而提高了内存访问的速度和效率。

* cudaHostAlloc()和cudaFreeHost()分配和释放分页锁定主机存储器；
* cudaHostRegister()分页锁定一段使用malloc()分配的存储器。

使用分页锁定主机存储器有许多优点：

* 在某些设备上，设备存储器和分页锁定主机存储器间数据拷贝可与内核执行并发进行；
* 在一些设备上，分页锁定主机内存可映射到设备地址空间，减少了和设备间的数据拷贝
* 在有前端总线的系统上，如果主机存储器是分页锁定的，主机存储器和设备存储器间的带宽会高些

然而分页锁定主机存储器是稀缺资源，所以分页锁定主机存储器的分配会比可分页内存分配早失败。
另外由于减少了系统可分页的物理存储器数量，分配太多的分页锁定内存会降低系统的整体性能。


可分享存储器
~~~~~~~~~~~~~~~~~~~~~~~~
可分享存储器(portable memory)。

一块分页锁定存储器可被系统中的所有设备使用，但是默认的情况下，上面说的使用分页锁定存储器的好处只有分配它时，
正在使用的设备可以享有（如果可能的话，所有的设备共享同一个地址空间）。
为了让所有设备都可以享有分页锁定存储器的好处，可以在使用cudaHostAlloc()分配时传入cudaHostAllocPortable标签，
或者在使用cudaHostRegister()分页锁定存储器时，传入cudaHostRegisterPortable标签。

写结合存储器
~~~~~~~~~~~~~~~~~~~~~~~~~~
默认情况下，分页锁定主机存储器是可缓存的。可以在使用cudaHostAlloc()分配时传入cudaHostAllocWriteCombined标签使其被分配为写结合的。
写结合存储器不占用主机的一级和二级缓存资源，所以应用的其它部分就有更多的缓存可用。
另外写结合存储器在通过PCI-e总线传输时不会被监视（snoop），这能够获得高达40%的传输加速。

从主机读取写结合存储器极其慢，所以写结合存储器应当只用于那些主机只写的存储器。

被映射存储器
~~~~~~~~~~~~~~~~~~~~
在一些设备上，在使用cudaHostAlloc()分配时传入cudaHostAllocMapped标签
或者在使用cudaHostRegister()分页锁定一块主机存储器时使用cudaHostRegisterMapped标签，
可分配一块被映射到设备地址空间的分页锁定主机存储器。

这块存储器有两个地址：一个在主机存储器上，一个在设备存储器上。
主机指针是从cudaHostAlloc（）或malloc()返回的， 设备指针可通过cudaHostGetDevicePointer()函数检索到，
可以使用这个设备指针在内核中访问这块存储器。

唯一的例外是主机和设备使用统一虚拟地址空间(UVA)时，此时主机指针和设备指针相同，可以直接在内核中使用主机指针。

从内核中直接访问主机存储器有许多优点：

* 无须在设备上分配存储器，也不用在这块存储器和主机存储器间显式传输数据；数据传输是在内核需要的时候隐式进行的。
* 无须使用流重叠数据传输和内核执行；数据传输和内核执行自动重叠。

由于被映射分页锁定存储器在主机和设备间共享，应用必须使用流或事件来同步存储器访问以避免任何潜在的读后写，写后读，或写后写危害。

为了在给定的主机线程中能够检索到被映射分页锁定存储器的设备指针，必须在调用任何CUDA运行时函数前调用cudaSetDeviceFlags()，
并传入cudaDeviceMapHost标签。否则，cudaHostGetDevicePointer()将会返回错误。
（在支持UVA的64位系统上，映射功能默认启用，不需要再设置这个标签。）

如果设备不支持被映射分页锁定存储器，cudaHostGetDevicePointer()将会返回错误。
应用可以检查canMapHostMemory属性应用以查询这种能力，如果支持映射分页锁定主机存储器，将会返回1。

注意：从主机和其它设备的角度看，操作被映射分页锁定存储器的原子函数（原子函数节）不是原子的。

异步并发执行
`````````````````````
为了易于使用主机和设备间的异步执行，一些函数是异步的;在设备完全完成任务前，控制已经返回给主机线程了。它们是：

* 内核发射；
* 设备内两个不同地址间的存储器拷贝函数；
* 从主机到设备拷贝不超过64KB的存储器块（源数据是可分页内存时，函数会在数据拷贝到暂存缓冲区后返回）；
* 存储器拷贝函数中带有Async后缀的；
* 设置设备存储器的函数调用。

程序员可通过将 ``CUDA_LAUNCH_BLOCKING`` 环境变量设置为1来全局禁用所有运行在系统上的应用的异步内核发射。
提供这个特性只是为了调试，永远不能作为使软件产品运行得可靠的方式。

在下面的情形中，内核启动是同步的：

* 通过性能剖析器（如Nsight Compute）收集硬件计数器时。

**数据传输和内核执行重叠:** 设备可在内核执行时，在分页锁定存储器和设备存储器之间拷贝数据。

应用可以通过检查asyncEngineCount 设备属性查询这种能力，如果其大于0，说明设备支持数据传输和内核执行重叠。

**并发内核执行：** 计算能力2.0及以上的设备可并发执行多个内核。应用可以检查concurrentKernels属性以查询这种能力，如果等于1，说明支持。
最大可并发执行的内核数目取决于计算能力（例如计算能力3.5为32，现代GPU为128），具体见编程指南中的计算能力表。

来自不同CUDA上下文的内核不能并发执行。使用了许多纹理或大量本地存储器的内核和其它内核并发执行的可能性比较小。

**并发数据传输：** 在有两个拷贝引擎的设备上，从主机分页锁定存储器复制数据到设备存储器和从设备存储器复制数据到主机分页锁定存储器，
这两个操作可并发执行。应用可以通过检查asyncEngineCount 属性查询这种能力，如果等于2，说明支持。

流(stream)
~~~~~~~~~~~~~~~~~~
应用通过流管理并发。流是一系列顺序执行的命令（可能是不同的主机线程发射）。
另外，不同流之间相对无序的或并发的执行它们的命令；这种行为是没有保证的，而且不能作为正确性的的保证（如内核间的通信没有定义）。

**默认流:** 没有使用流参数的内核启动和主机设备间数据拷贝，或者等价地将流参数设为0，此时发射到默认流。因此它们顺序执行。
默认情况下默认流是legacy默认流，会与同一设备上其他所有阻塞流同步；
编译时加上 ``--default-stream per-thread`` 后，每个主机线程有各自的默认流，并且不再与其他流隐式同步。

创建和销毁 
''''''''''''''''''''''
可以通过创建流对象来定义流，且可指定它作为一系列内核发射和设备主机间存储器拷贝的流参数。

* CUDA 流的创建和销毁都是在主机端进行的，因此不需要在设备端分配内存。
* CUDA 流的创建和销毁是一个比较耗时的操作，因此在使用时需要谨慎分配和使用。

.. code-block:: cpp

    // 创建一个 CUDA 流
    cudaError_t cudaStreamCreate(cudaStream_t *pStream);
    // pStream 是指向 CUDA 流的指针。
    // 如果创建 CUDA 流成功，cudaStreamCreate()函数将返回cudaSuccess。如果创建失败，将返回相应的错误码。
    cudaError_t cudaStreamDestroy(cudaStream_t stream);

.. literalinclude:: code/example6.cu
    :language: cu

调用cudaStreamDestroy()时，如果流中还有未完成的任务，函数会立即返回，等设备完成流中的所有任务后再自动释放流的资源。

显式同步
''''''''''''''''''''''

.. code-block:: cpp

    // 等待设备端所有操作完成。会阻塞主机端程序的执行，直到设备端所有操作都完成。
    cudaError_t cudaDeviceSynchronize(void);

    // 等待指定的 CUDA 流中的所有操作完成
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);

    // 让stream中之后提交的任务等待event完成后再执行。不会阻塞主机端，常用于不同流（甚至不同设备）之间的同步。
    cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);
    // stream 是需要等待的 CUDA 流，
    // event 是被等待的 CUDA 事件，
    // flags 是等待标志：cudaEventWaitDefault（默认）或 cudaEventWaitExternal。

    // 等待指定的 CUDA 事件完成。会阻塞主机端程序的执行，直到指定的 CUDA 事件完成。
    cudaError_t cudaEventSynchronize(cudaEvent_t event);

    // 查询指定的 CUDA 流中的操作是否完成
    cudaError_t cudaStreamQuery(cudaStream_t stream);
    // 如果指定的 CUDA 流中的操作已经完成，cudaStreamQuery() 函数将返回cudaSuccess。
    // 如果指定的 CUDA 流中的操作还没有完成，cudaStreamQuery() 函数将返回 cudaErrorNotReady。

为了避免不必要的性能损失，这些函数最好用于计时或隔离失败的发射或存储器拷贝。

隐式同步
''''''''''''''''''''

如果是下面中的任何一种操作在来自不同流的两个命令之间，这两个命令也不能并发：

* 分页锁定主机存储器分配，
* 设备存储器分配，
* 设备存储器设置，
* 设备内两个不同地址间的存储器拷贝函数；
* 默认流中调用的任何CUDA命令

回调
''''''''''''''''''
运行时通过cudaLaunchHostFunc()（旧接口为cudaStreamAddCallback()）提供了一种在任何执行点向流插入回调的方式。
回调是一个函数，一旦在插入点之前发射到流的所有命令执行完成，回调就会在主机上执行。
在流0中的回调，只能在插入点之前其它流的所有命令都完成后才能执行。

回调函数中不能直接或间接地调用任何CUDA API，否则可能导致错误或死锁。回调执行完之前，流中后续的命令不会开始执行。

.. code-block:: cpp

    // 推荐使用的接口
    cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
    // fn 的类型为 void (*)(void *userData)

    // 旧接口
    cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, 
                                                void *userData, unsigned int flags);
    // stream 是要添加回调函数的 CUDA 流
    // callback 的类型为 void (*)(cudaStream_t stream, cudaError_t status, void *userData)
    // userData 是传递给回调函数的用户数据
    // flags 是保留参数，必须为0

.. literalinclude:: code/example7.cu
    :language: cu


流的优先级
''''''''''''''''''''''''''
默认情况下，所有的CUDA流具有相同的优先级。
除此之外，CUDA流也可以指定优先级：当有线程块等待调度时，GPU会优先调度高优先级流中内核的线程块。
有了这个功能，我们可以让时间紧迫的GPU操作优先执行。

.. code-block:: cpp

    // 获取设备支持的CUDA流优先级范围
    // leastPriority和greatestPriority分别表示设备支持的最小和最大CUDA流优先级
    cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

    // 创建一个具有指定优先级的CUDA流,但是需要注意优先级的范围。
    // priority值越小，优先级越高
    cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
    // 一些常用的flags值：
    // cudaStreamDefault：默认标志，表示创建一个CUDA流，并和默认流同步。
    // cudaStreamNonBlocking：非阻塞标志，表示创建一个非阻塞的CUDA流，和默认流并行执行

    // 判断是否支持流优先级，compute capability 3.5及以上都支持
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    if (prop.streamPrioritiesSupported == 0){
        // 不支持
    }

.. note:: 

    流优先级只影响内核线程块的调度顺序，不会改变数据传输的顺序，也不会抢占正在执行的线程块



事件
~~~~~~~~~~~~
通过在应用的任意点上异步地记载事件和查询事件是否完成，运行时提供了精密地监测设备运行进度和精确计时。

当事件记载点前面，事件指定的流中的所有任务或者指定流中的命令全部完成时，事件被记载。
只有记载点之前所有的流中的任务/命令都已完成，0号流的事件才会记载。

.. code-block:: cpp

    // 创建一个 CUDA 事件
    cudaError_t cudaEventCreate(cudaEvent_t *event);

    // 销毁指定的 CUDA 事件
    cudaError_t cudaEventDestroy(cudaEvent_t event);

    // 记录指定的 CUDA 事件
    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);

    // 计算两个 CUDA 事件之间的时间差，单位为毫秒，分辨率约0.5微秒
    cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

    // 等待一个CUDA事件的完成,会阻塞程序的执行,直到指定的事件完成为止。
    cudaError_t cudaEventSynchronize(cudaEvent_t event);

.. literalinclude:: code/example8.cu
    :language: cu


CUDA dynamic parallelism(动态并行)
````````````````````````````````````````
CUDA dynamic parallelism (CDP)是一个设备运行时的功能，允许在内核中启动新的内核（子网格），计算能力3.5及以上支持。
子网格可以使用与父网格不同的网格和块大小，当问题的不同部分需要不同的并行度时，这个功能很有用。
使用时需要以 ``-rdc=true`` 编译并链接cudadevrt库。

动态并行的好处之一是，我们可以创建一个递归。

Grid-level cooperative groups
`````````````````````````````````````
有两个网格级合作组：grid_group和multi_grid_group。
使用这些组，程序员可以描述网格的操作以在单个GPU或多个GPU上进行同步。
使用grid_group进行网格级同步时，内核必须通过cudaLaunchCooperativeKernel()启动。
multi_grid_group（多GPU同步）已在CUDA 11中被废弃，不建议再使用。

多进程服务（MPS）模式
`````````````````````````````
GPU能够执行来自并发的CPU进程的内核。然而，默认情况下，它们只是以时间分割的方式执行，即使每个内核并没有完全利用GPU计算资源。

为了解决这种不必要的串行，GPU提供了多进程服务（MPS）模式，多个进程的内核可以通过同一个MPS服务器真正并发地在GPU上执行。

使用以下命令进行设置（将计算模式设为EXCLUSIVE_PROCESS是推荐做法，不是必需的）

.. code-block:: shell

    nvidia-smi -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d
    # 关闭MPS
    echo quit | nvidia-cuda-mps-control


Message Passing Interface (MPI)
`````````````````````````````````````````
消息传递接口（MPI）是一个并行计算接口标准，它可以在多个计算单元（CPU核心、GPU和节点）上启动多个进程，进程之间通过消息传递通信。
支持CUDA的MPI实现（CUDA-aware MPI）可以直接传递设备内存指针。

MPI库有OpenMPI, MVAPICH, Intel MPI

https://hpc-tutorials.llnl.gov/mpi/

.. code-block:: shell

    wget -O /tmp/openmpi-3.0.4.tar.gz \
           https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.4.tar.gz
    tar xzf /tmp/openmpi-3.0.4.tar.gz -C /tmp
    cd /tmp/openmpi-3.0.4
    ./configure --enable-orterun-prefix-by-default --with-cuda=/usr/local/cuda
    make -j $(nproc) all && sudo make install
    sudo ldconfig
    mpirun --version


混合精度计算
-----------------------
半精度计算
`````````````````````
cuda为fp16提供了内置命令，同时还提供了数据类型转换函数

.. literalinclude:: code/mixed_precision_half.cu
    :language: cu


int8、int16操作
`````````````````````````````
对于8位/16位整数，CUDA提供了矢量的点乘操作。这些操作是DP4A（带累加的四元点乘）和DP2A（带累加的二元点乘），计算能力6.1及以上支持。
使用它，你可以编写只有8位或8位/16位的混合运算，并进行32位的整数累加。

.. literalinclude:: code/mixed_precision_int.cu
    :language: cu



多设备系统
-----------------------
主机系统上可以有多个设备。下面的代码展示了怎样枚举这些设备、查询它们的属性、确定有多少个支持CUDA的设备。

.. literalinclude:: code/cudadevice1.cpp
    :language: cpp


GPUDirect-gpu直连
```````````````````````````````
GPUDirect在各个显卡之间创建高带宽、低延迟的连接。

GPUDirect可以被分为几类：

* Peer to peer (P2P) transfer between GPU：允许CUDA程序使用高速的直接内存访问(DMA, Direct Memory Access),在同一系统中的两个GPU之间复制数据。
  它还允许优化对同一系统中其他GPU的内存访问。
* Accelerated communication between network and storage(加速网络和存储之间的通信):这项技术有助于从第三方设备直接访问CUDA内存,
  如InfiniBand网络适配器或存储的直接访问。它消除了不必要的内存复制和CPU开销，从而减少了传输和访问的延迟。该功能从CUDA 3.1开始支持。
* GPUDirect for video:这项技术为基于帧的视频设备优化了管道。它允许与OpenGL、DirectX或CUDA进行低延迟的通信。并从CUDA 4.2开始支持。
* Remote Direct Memory Access (RDMA):这一功能允许集群中不同节点的GPU之间通过网卡直接通信，不经过主机内存。该功能从CUDA 5.0开始支持。


设备指定
```````````````````````
在任何时候，主机线程都可以使用cudaSetDevice()来设置它操作的设备。
设备存储器分配和内核执行都作用在当前的设备上；流和事件关联当前设备。
如果没有cudaSetDevice()调用，当前设备为0号设备。

* 如果内核执行和存储器拷贝发射到非关联到当前设备的流，它们将会失败。
* 如果输入事件和输入流关联到不同的设备，cudaEventRecord()将失败。
* 如果两个输入事件关联到不同的设备，cudaEventElapsedTime()将会失败。
* 即使输入事件关联的设备并非当前设备，cudaEventSynchronize()和cudaEventQuery()也会成功。
* 即使输入流和输入事件关联到不同的设备，cudaStreamWaitEvent()也会成功。因此cudaStreamWaitEvent()可用于在不同的设备同步彼此。
* 每个设备有自己的默认流，因此在一个设备上发射到默认流的一个命令会和发射到另一个设备上默认流中的命令并发执行。


p2p存储器
```````````````````````
在NCCL中禁用P2P: ``export NCCL_P2P_DISABLE=1`` （只影响NCCL的通信方式，不会禁用CUDA本身的P2P功能）


**存储器访问**

计算能力2.0或以上的设备（64位应用中）能够访问彼此的存储器（即运行在一个设备上的内核可以解引用指向另一个设备存储器的指针）。
只要两个设备上的cudaDeviceCanAccessPeer()返回true，这种p2p的存储器访问特性在它们间得到支持。
但必须通过调用cudaDeviceEnablePeerAccess()启用两个设备间的p2p存储器访问支持。

.. literalinclude:: code/cudadevice2.cu
    :language: cu

**存储器复制**

可以在两个不同设备间的存储器上复制存储器内容。
当两个设备使用统一存储器地址空间时，使用设备存储器节提到的普通的存储器拷贝函数即可。
否则使用cudaMemcpyPeer()、cudaMemcpyPeerAsync()、cudaMemcpy3DPeer()或者cudaMemcpy3DPeerAsync()

两个不同设备之间的存储器复制（使用默认流时）：

* 直到前面发射到任何一个设备的命令执行完，才开始执行
* 只有在它们执行完之后，后面发射到两者中任一设备的异步命令可开始。

注意如果通过如p2p存储器访问节描述的cudaDeviceEnablePeerAccess()启用两
个设备间的p2p访问，两个设备间的p2p存储器拷贝就没有必要通过主机进行，因此更快。





计算模式
------------------
使用NVIDIA的系统管理接口（nvidia-smi）可以设置系统上任何设备的计算模式，nvidia-smi是一个作为驱动一部分发布的工具。

nvidia-smi -c mode

0/DEFAULT, 1/EXCLUSIVE_THREAD (DEPRECATED),2/PROHIBITED, 3/EXCLUSIVE_PROCESS

目前可用的是下面三种：

* 默认模式(DEFAULT)：多个主机线程和多个进程可同时使用设备（使用运行时调用cudaSetDevice()，或使用驱动API时将关联到设备的上下文作为当前上下文）。
* 独占进程模式(EXCLUSIVE_PROCESS)：在系统的所有进程之间，一个设备上只能建立一个CUDA上下文，该上下文可以成为建立该上下文的进程中的许多线程的当前上下文，可从多个线程同时使用。
* 禁止模式(PROHIBITED)：设备上不允许建立任何CUDA上下文（不允许运行计算应用程序）。

已废弃的独占线程模式(EXCLUSIVE_THREAD)要求一个设备上只能建立一个CUDA上下文，而且一次只能成为一个线程的当前上下文。


SIMT架构
-------------
多处理器以32个为一组创建、管理、调度和执行并行线程，这32个线程称为束（warps）。
束内包含的不同线程从同一程序地址开始，但它们有自己的指令地址计数器和寄存器状态，因此可自由分支和独立执行。

束这个术语来源于纺织（weaving）这种并行线程技术。半束（half-warp）是束的前一半或后一半。
四分之一束是指第一、第二、第三或第四个束的四分之一。

当多处理器得到一个或多个块执行，它会将块分割成束以执行，束被束调度器调度。
块分割成束的方式总是相同的；束内线程是连续的，递增线程ID，第一个束包含线程0。

束每次执行一个相同的指令，所以如果束内所有32个线程在同一条路径上执行的话，会达到最高效率。
如果由于数据依赖条件分支导致束分岔，束会顺序执行每个分支路径，而禁用不在此路径上的线程，直到所有路径完成，
线程重新汇合到同一执行路径。分支岔开只会在同一束内发生；不同的束独立执行不管它们是执行相同或不同的代码路径。

在使用单指令控制多处理元素这点上，SIMT架构类似SIMD（单指令，多数据）向量组织方法。
重要的不同在于SIMD组织方法会向应用暴露SIMD宽度，而SIMT指定单线程的执行和分支行为。
与SIMD向量机相反，SIMT允许程序员为独立标量线程编写线程级并行代码，也为协作线程编写数据并行代码。
为了正确性，程序员可忽略SIMT行为；只要维护束内线程很少分支的代码就可显著提升性能。
实践中，这类似于传统代码中缓存线的角色：以正确性为目标进行设计时，可忽略缓存线尺寸，但如果以峰值性能为目标进行设计，
在代码结构中就必须考虑。另外，向量架构要求软件将负载合并成向量，并手动管理分支。

如果一个束执行非原子指令为多个线程写入全局存储器或共享存储器的同一位置，串行写入该位置变量的数目依赖于设备的计算能力且那个线程最后写入无法确定。

如果束执行的原子指令为束内多个线程读、修改和写入全局存储器的同一位置，每次读、修改和写入都会串行执行，但是他们执行的顺序没有定义。

.. note::

    从Volta架构开始引入了独立线程调度，束内每个线程都有自己的程序计数器，分岔后的线程可以交错执行，
    不能再假设束内线程是隐式同步的，需要同步时应使用__syncwarp()或带_sync后缀的warp原语。




cuda编程库
----------------------------
cuda编程库提供了经过优化的各种方法，而不需要自己去实现并优化kernels

| cuBLAS:基础线性代数库
| cuFFT:快速傅里叶变换库
| cuRAND:随机数生成库
| cuDNN:专门用于优化深度学习
| NPP:图像和信号处理
| cuSPARSE:稀疏的线性代数库
| cuSolver:LAPACK in GPU
| nvGRAPH:图分析库（已在CUDA 11中移除）
| Thrust:STL in CUDA

cuBLAS
`````````````````
* level-1:向量乘向量
* level-2:矩阵乘向量
* level-3:矩阵乘矩阵

.. code-block:: cpp

    cublasHandle_t handle;
    cublasCreate(&handle);
    // .. { data operation } ..
    cublasSgemm(...);
    // .. { data operation } ..
    cublasDestroy(handle);

**多GPU**

.. code-block:: cpp

    cublasXtHandle_t handle;
    cublasXtCreate(&handle);
    int num_of_devices;
    cudaGetDeviceCount(&num_of_devices);
    int *devices = (int *)calloc(num_of_devices, sizeof(int));
    for (int i = 0; i < num_of_devices; i++)
        devices[i] = i;
    cublasXtDeviceSelect(handle, num_of_devices, devices);
    cublasXtSgemm( ... );
    cublasXtDestroy(handle);
    free(devices);


**混合精度计算**

.. code-block:: cpp

    cublasGemmEx(...);

混合精度计算时，可以使用TensorCore进行加速。从CUDA 11开始，只要数据类型和计算类型(cublasComputeType_t)允许，
cuBLAS默认就会使用TensorCore，algo参数传CUBLAS_GEMM_DEFAULT即可；早期版本使用的CUBLAS_GEMM_DEFAULT_TENSOR_OP已被废弃。


cuRAND
```````````````````````
随机数生成

.. code-block:: cpp

    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);
    curandGenerate();
    curandGenerateUniform(curand_gen, random_numbers_on_device_memory, length);

cuFFT
```````````````
.. code-block:: cpp

    cufftPlan1d();
    cufftPlan2d();
    cufftPlan3d();
    // If sample data has a batched and stride layout
    cufftPlanMany();
    // sample size is greater than 4 GB
    cufftMakePlanMany64();
    cufftXtMakePlanMany();
    cufftCreate();
    // Complex-to-complex (double: cufftExecZ2Z)
    cufftExecC2C();
    // Real-to-complex (double: cufftExecD2Z)
    cufftExecR2C();
    // Complex-to-real (double: cufftExecZ2D)
    cufftExecC2R();


cuDNN
`````````````````````
.. image:: /images/cuda编程/cudnn_tensorrt.jpg

api: https://docs.nvidia.com/deeplearning/cudnn/api/index.html

* cuDNN激活函数：sigmoid, ReLU, tanh, clipped ReLU, ELU, swish(cuDNN 8的较新版本加入) 和 identity（identity只能用于cudnnConvolutionBiasActivationForward）。
  还提供了cudnnActivationForward()和cudnnActivationBackward()进行前向和反向计算
* cudnnSoftmaxForward
* cudnnConvolutionForward()、cudnnAddTensor()；cudnnConvolutionBackwardData()、cudnnConvolutionBackwardFilter()、cudnnConvolutionBackwardBias();
  cudnnGetConvolution2dForwardOutputDim()、cudnnGetConvolutionForwardAlgorithm_v7()、cudnnFindConvolutionForwardAlgorithm()、cudnnGetConvolutionForwardWorkspaceSize、
  cudnnGetConvolutionBackwardDataAlgorithm_v7、cudnnGetConvolutionBackwardDataWorkspaceSize、cudnnGetConvolutionBackwardFilterAlgorithm_v7、
  cudnnGetConvolutionBackwardFilterWorkspaceSize
  （不带_v7后缀的cudnnGetConvolution*Algorithm()已在cuDNN 8中移除）
* cudnnCreatePoolingDescriptor、cudnnSetPooling2dDescriptor、cudnnGetPooling2dForwardOutputDim、
  cudnnPoolingForward、cudnnPoolingBackward
* cuDNN支持4种RNN，RNN with ReLU,RNN with tanh, LSTM, and GRU

  - cuDNN 8及以后使用cudnnSetRNNDescriptor_v8()、cudnnGetRNNWeightSpaceSize()、cudnnRNNForward()、cudnnRNNBackwardData_v8()、cudnnRNNBackwardWeights_v8()
  - cudnnRNNForwardInference()、cudnnRNNForwardTraining()、cudnnSetRNNDescriptor_v6()、cudnnGetRNNParamsSize()是旧接口，在cuDNN 8中被废弃，cuDNN 9中已移除


cuDNN推理视频：https://developer.nvidia.com/gtc/2019/video/S9644/video
  




OpenACC
-----------------------------
OpenACC是一种并行计算编程模型，用于在加速器（如GPU）上开发高性能计算应用程序。
它是一个开放的标准，由多家公司和研究机构共同开发和维护，旨在简化并行计算的编程过程。

通过OpenACC，开发人员可以使用指令将并行计算任务分配给加速器，并利用其强大的并行处理能力来加速应用程序的执行。
OpenACC提供了一组预定义的指令，开发人员可以将其插入到现有的串行代码中，以指示哪些部分可以并行执行。
这些指令指导编译器生成适合加速器的并行代码。

OpenACC支持多种编程语言，包括C、C++和Fortran，并且可以与其他并行编程模型（如OpenMP）结合使用。
它被广泛应用于科学计算领域，特别是在需要处理大规模数据集和复杂算法的应用程序中，以提高计算性能和效率。





https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese