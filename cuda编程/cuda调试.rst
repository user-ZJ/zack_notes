cuda调试
=====================
NVPROF 和 NVVP英伟达公司自己开发和维护的剖析工具，作为CUDA的一部分提供。

* NVPROF：NVIDIA Profiler(命令行工具)
* NVVP：NVIDIA Visual Profiler(有图形界面)

环境配置
-----------------------
.. code-block:: shell

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/


剖析GPU应用中的重点目标范围
--------------------------------------
剖析目标可以是特定的代码块、GPU和time。指定代码块的做法被称为重点剖析。
这种技术在你想集中分析一个特定的内核函数，或对大型GPU应用程序的部分进行分析时，这种技术非常有用。

在目标代码中添加以下代码：

.. code-block:: cpp

    #include <cuda_profiler_api.h>
    cudaProfilerStart();
    // ... {target of profile} ...
    cudaProfilerStop();

生成profile文件：

.. code-block:: shell

    nvcc -m64 -gencode arch=compute_70,code=sm_70 -o sgemm sgemm.cu
    nvprof -f -o profile-start-stop.nvvp --profile-from-start off ./sgemm

CUDA程序运行时间统计
---------------------------------
.. code-block:: cpp

    // Initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    //... Execution code ...
    // Getting elapsed time
    cudaDeviceSynchronize(); // Blocks the host until GPU finishes the work
    sdkStopTimer(&timer);
    // Getting execution time in micro-secondes
    float execution_time_ms = sdkGetTimerValue(&timer)
    // Termination of timer
    sdkDeleteTimer(&timer);


NVIDIA Tools Extension (NVTX).
-------------------------------------------
.. code-block:: cpp

    #include "nvToolsExt.h"
    cudaProfileStart();
    nvtxRangePushA("Annotation");
    // Range of GPU operations
    cudaDeviceSynchronization();
    nvtxRangePop();
    cudaProfileStop();

.. code-block:: shell

    nvcc -m64 -gencode arch=compute_70,code=sm_70 -lnvToolsExt -o sgemm sgemm.cu
    nvprof -f --profile-from-start off -o sgemm.nvvp ./sgemm


NVIDIA Visual profiling
--------------------------------------------
.. warning:: 

    NVIDIA Visual profiling不适用于计算能力在7.2以上的显卡，计算能力在7.2以上的显卡推荐使用NsightCompute

.. tip::

    linux下，使用sudo nvvp来启动图形界面 

NVIDIA Visual profiling支持的调试方式有：

1. 本机调试
2. 使用nvprof命令生成性能分析文件，再拷贝到主机上调试
3. 连接远程服务器调试

.. figure:: /images/cuda编程/visualProfile5.jpg

    NVIDIA Visual profiling界面

NVIDIA Visual profiling命令行调试
`````````````````````````````````````````````````
.. code-block:: shell

    # 直接打印出各个阶段执行时间
    nvprof xxx.bin 

* --print-gpu-trace

NVIDIA Visual profiling远程机器可视化
`````````````````````````````````````````````
连接到远程服务器调试步骤如下：

1. File -> New session

.. image:: /images/cuda编程/visualProfile1.jpg

2. 创建connection

.. image:: /images/cuda编程/visualProfile2.jpg

3. 配置远程cuda工具链路径

.. image:: /images/cuda编程/visualProfile3.jpg

4. 选择cuda程序的可执行文件(可指定执行参数)，然后选择下一步
5. 选择需要监测的性能选项

.. image:: /images/cuda编程/visualProfile4.jpg

错误解决
`````````````````````
1. nvprof --analysis-metrics报错

.. code-block:: shell

    Warning: ERR_NVGPUCTRPERM - The user does not have permission to profile on the target device. See the following link for 
    instructions to enable permissions and get more information: https://developer.nvidia.com/ERR_NVGPUCTRPERM

解决办法：

.. code-block:: shell

    sudo vim /etc/sudoers
    # 修改secure_path
    secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/usr/local/cuda/bin"
    #wq!保存




使用CUDA error调试CUDA程序
-----------------------------------
cudaGetLastError:用于获取最近一次调用CUDA API函数时发生的错误。

.. code-block:: cpp
    
    cudaError_t cudaGetLastError();

checkCudaError:用于检查CUDA错误,在common/inc/cuda_helper.h中定义

.. code-block:: cpp

    #define checkCudaErrors(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
        err, cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
        } \
    }
    #endif

    // 使用
    checkCudaErrors(cudaMalloc((void **)&d_A, N * K * sizeof(float)));

CUDA API使用checkCudaErrors，cuda kernel调用使用
checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaDeviceSynchronize());检查错误

设置CUDA_LAUNCH_BLOCKING=1环境变量，来是的所有kernel执行完成后强制和host同步，来达到检查错误的目的

CUDA断言
--------------------
断言是用来验证操作结果是否符合预期。断言函数可以从设备代码中调用，并且可以在给定参数为零时停止内核的执行。
如果应用程序被调试器启动，它将作为一个断点工作，这样开发者就可以对给定的信息进行调试。

.. code-block:: cpp

    void assert(int expression);

使用断言会对应用程序的性能产生影响。因此，我们应该只在调试时使用断言。
在生产环境中运行时，建议禁用它。你可以在编译时禁用断言，方法是添加NDEBUG预处理宏，然后再包含assert.h。

使用Nsight Visual Studio调试cuda程序
-----------------------------------------------
对于Windows应用程序开发人员，CUDA工具包提供了Nsight Visual Studio，它可以在Visual Studio中实现GPU计算。
这个工具作为Visual Studio的一个扩展，但你可以与主机一起构建、调试、剖析和跟踪GPU应用程序。

Nsight Eclipse调试CUDA程序
--------------------------------------
对于Linux和OSX平台的开发，CUDA工具箱提供了Nsight Eclipse。这个工具是基于Eclipse的，所以开发者可以很容易地在CUDA C开发中使用这个工具。

这个工具是作为一个软件包与CUDA工具包一起安装的，所以你不必单独安装这个工具。
然而，如果你使用的是Linux，需要为其操作配置Java 7


CUDA gdb调试
--------------------------
使用cuda gdb,主机代码编译时需要添加-g参数，GPU代码编译时需要添加-G参数。

**从内核代码切换到主机代码，需要使用continue指令**

cuda-gdb大部分调试命令和gdb相同


* help info cuda：列出所有info cuda指令
* info cuda kernels：列出所有kernel
* breakpoint kerne_name:像普通函数一样，CUDA-GDB可以在内核函数上设置断点
* help cuda：查看cuda指令
* cuda device kernel block thread:打印当前线程信息
* cuda thread：列出当前活跃的GPU线程（若有的话）；cuda thread(1, 1, 1)
* cuda kernel：列出当前活跃的GPU Kernel，并允许将“焦点”转移到指定的GPU线程
* info cuda指令：

  - info cuda devices:设备信息
  - info cuda sms:设备的流处理器信息
  - info cuda warps:在此SM上的warp信息
  - info cuda lanes:lanes的信息
  - info cuda kernels:当前核函数信息
  - info cuda blocks:当前blocks信息
  - info cuda threads:当前活动线程信息
  - info cuda launch trace:information about the parent kernels of the kernel in focus
  - info cuda launch children:information about the kernels launched by the kernels in focus
  - info cuda contexts:information about all the contexts


参考： https://huanghailiang.github.io/2018/07/02/2-CUDA-gdb/

.. note:: 

    cmake中编译Debug版本设置：

    set(CUDA_NVCC_FLAGS "-G;-g")

.. note:: 

    | 报错：cuda-gdb: error while loading shared libraries: libtinfo.so.5: cannot open shared object file: No such file or directory
    | 解决方法： sudo apt install libncurses*

CUDA-memcheck
---------------------------------
CUDA-memcheck是运行时检查GPU内存是否越界或非法访问等违规操作的工具。

要使用CUDA-memcheck，需要添加-Xcompiler -rdynamic编译参数。

CUDA-memcheck可以在cuda-gdb中使用，在cuda-gdb命令行通过set cuda memcheck on命令开启CUDA-memcheck

.. code-block:: shell

    cuda-memcheck [options] <application>

Nsight Systems
---------------------------
文档地址：https://docs.nvidia.com/nsight-systems/UserGuide/index.html

Nsight Systems是一个系统的性能分析工具，可以在时间轴上可视化操作，并轻松找到优化点。

Nsight Systems支持图形界面和命令行两种方式，本地调试使用图形界面，远程调试使用命令行生成性能分析数据，再用图像界面打开。

使用前需要配置环境变量：export PATH=$PATH:/usr/local/cuda-11.3/nsight-systems-2021.1.3/target-linux-x64

.. code-block:: shell

    nsys profile -t osrt,cuda,nvtx,cublas,cudnn -o baseline -w true \<command\>

所有命令行选项均区分大小写。对于命令开关选项，当使用短选项时，参数应跟在开关后面一个空格；例如-s 进程树。
当使用长选项时，开关后面应该跟一个等号，然后是参数；例如--sample=进程树。


* -t/--trace
  
  - cuda: For tracing CUDA operations
  - nvtx: For tracing nvtx tags
  - cublas, cudnn, opengl
  - openacc: For tracing the API operation
  - osrt: For tracing OS runtime libraries
  - none: No API trace
  
* -o/--output:Output filename
* -w/--show-output:true/false: Prints out the behavior of the profiler on the Terminal


示例：https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/using-nsight-system-to-realize-performance-analysis?spm=a2c4g.11186623.help-menu-85222.d_3_0_1.45bd1f3abzMt5H

Nsight Compute
--------------------
文档地址：https://docs.nvidia.com/nsight-compute/index.html

Nsight Compute是一个kernel级别的分析工具。它收集了GPU的指标信息，并帮助我们专注于CUDA内核的优化。

Nsight Compute同样支持图形界面和命令行

使用前需要配置环境变量：export PATH=$PATH:/usr/local/cuda-11.3/nsight-compute-2021.1.0

.. code-block:: shell

    nv-nsight-cu-cli -o <output filename> <application command>

* 如果不是用-o,性能分析信息会打印在串口
* --kernel-regex: Specifies the kernel to the profile
* --devices: Focuses on profiling a specific GPU




nvidia-smi
-----------------------
.. image:: /images/cuda编程/nvidia-smi.jpg

* 第一行报告驱动程序版本和支持的CUDA版本
* 第二行显示GPU统计格式
* 每个连续行包含每个GPU的统计数据，包括以下内容：
  
  - GPU ID
  - Operation mode:
  
    + Persistence mode (ON/OFF)
    + Tesla Compute Cluster (TCC)/Windows Display
     
  - Driver Model (WDDM) mode
  - Fan speed
  - GPU temperature
  - Performance mode
  - Power usage and capacity
  - Bus-ID
  - Memory usage and installed memory
  - Counted error-correcting code (ECC)
  - GPU utilization
  - Compute mode

nvidia-smi命令选项：

* -i,--id:根据显卡编号指定显卡
* -l,--loop:以指定的时间间隔报告GPU的状态
* -f,--filename:将信息写到文件
* -L:列出所有可用的 NVIDIA 设备
* -q,--query:结构化输出报告,可以指定利用率、功耗、内存和时钟速度统计等信息
  
  - nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,
    pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,
    utilization.memory,memory.used,memory.free,memory.used --format=csv -l 1
  - nvidia-smi --query-gpu=index,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,
    clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,
    clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,
    clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sync_boost --format=csv

* -pl=N:以瓦特为单位指定最大功率管理限制。


* nvidia-smi -q -d CLOCK:查看当前的 GPU 时钟速度、默认时钟速度和最大可能的时钟速度
* nvidia-smi -q -d SUPPORTED_CLOCKS :显示每个 GPU 的可用时钟速度列表

监控GPU
```````````````````````
**nvidia-smi dmon -s pucvmet -i 0**

监控GPU0的使用信息

参数说明：

* p: Power usage and temperature
* u: Utilization
* c: Proc and mem clocks
* v: Power and thermal violations
* m: FB and Bar1 memory
* e: ECC errors and PCIe replay errors
* t: PCIe Rx and Tx throughput

**nvidia-smi pmon -i 0 -s u -o T**

将空GPU0上不同进程使用GPU的信息

* sm%: CUDA core utilization
* mem%: Sampled time ratio for memory operations
* enc%/dec%: HW encoder's utilization
* fb: FB memory usage

查看多GPU的连接拓扑结构
`````````````````````````````````
* nvidia-smi topo -m
* nvidia-smi topo -p2p rwnap:查看GPU之间的p2p能力