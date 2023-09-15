cublas使用
========================
api: https://docs.nvidia.com/cuda/cublas/#using-the-cublas-api

cublasHandle_t：cublasHandle_t是cuBLAS库中的一个类型，用于表示cuBLAS上下文。
cuBLAS上下文是一个线程本地的对象，用于存储cuBLAS库的状态和资源，
例如cuBLAS库的版本、当前设备等。
在使用cuBLAS进行矩阵乘法运算时，需要创建一个或多个cuBLAS上下文，并在每个上下文中执行矩阵乘法操作。

* cuBLAS API:应用程序必须在GPU内存空间中分配所需的矩阵和向量，用数据填充它们，调用所需的cuBLAS函数序列，然后将结果从GPU内存空间拷贝回主机
* cuBLASXt API：多GPU调用的api，数据可以在显存或内存，运算完成后，会将数据拷贝到Host内存
* cuBLASLt：一个轻量级库，专用于通用矩阵到矩阵乘法 (GEMM) 运算，具有新的灵活 API。
  该库增加了矩阵数据布局、输入类型、计算类型以及通过参数可编程性选择算法实现和启发式的灵活性。
  用户识别出一组用于预期 GEMM 操作的选项后，这些选项可以针对不同的输入重复使用

cuBLAS API
--------------------
cuBLAS 库上下文的句柄(cublasHandle_t):

1. cublasCreate()创建，cublasDestroy()释放
2. cublasHandle_t保存设备相关的信息
3. 要使用不同的GPU，需要先cudaSetDevice()设置设备后，在使用cublasCreate()创建cublasHandle_t关联相关设备
4. cuBLAS 库上下文与cublasCreate()调用时当前的 CUDA 上下文紧密耦合。
   使用多个 CUDA 上下文的应用程序需要为每个 CUDA 上下文创建一个 cuBLAS 上下文，并确保前者的寿命永远不会超过后者。

线程安全
`````````````````
该库是线程安全的，即使使用相同的句柄，也可以从多个主机线程调用其函数。
当多个线程共享同一句柄时，更改句柄配置时需要格外小心，因为该更改可能会影响所有线程中后续的 cuBLAS 调用。
对于手柄的销毁更是如此。所以不建议多个线程共享同一个cuBLAS句柄。

结果再现性
`````````````````
给定工具包版本中的所有 cuBLAS API 例程在具有相同架构和相同 SM 数量的 GPU 上执行时，每次运行都会生成相同的按位结果。
但是，跨工具包版本不能保证按位再现性，因为某些实现更改可能会导致实现不同。

仅当单个 CUDA 流处于活动状态时，此保证才有效。如果多个并发流处于活动状态，则库可以通过选择不同的内部实现来优化总体性能。
多流执行的非确定性行为是由于在为并行流中运行的例程选择内部工作空间时进行了库优化所致。

为了避免这种影响，用户可以：

1. 使用cublasSetWorkspace()函数为每个使用的流提供单独的工作区
2. 每个流有一个 cuBLAS 句柄
3. 使用cublasLtMatmul()代替 GEMM 系列函数并提供用户拥有的工作区
4. 将调试环境变量CUBLAS_WORKSPACE_CONFIG设置为:16:8（可能会限制整体性能）或:4096:8（将使GPU内存中的库占用空间增加大约 24MiB）


cublas类型
----------------------

cublasStatus_t
```````````````````````
该类型用于函数状态返回。所有 cuBLAS 库函数都会返回其状态，该状态可以具有以下值。

+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| 值                             | 说明                                                                                                      |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_SUCCESS          | 操作成功                                                                                                  |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_NOT_INITIALIZED  | cuBLAS 库未初始化。                                                                                       |
|                                | 这通常是由于缺少先前的cublasCreate()调用、cuBLAS例程调用的CUDA运行时API中的错误或硬件设置中的错误引起的。 |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_ALLOC_FAILED     | cuBLAS 库内部资源分配失败。这通常是由cudaMalloc()故障引起的。                                             |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_INVALID_VALUE    | 不支持的值或参数传递给函数（例如负向量大小）。                                                            |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_ARCH_MISMATCH    | 该功能需要设备架构中缺少的功能；通常是由于计算能力低于5.0引起的。                                         |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_MAPPING_ERROR    | 访问GPU内存空间失败，通常是由于纹理绑定失败造成的。                                                       |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_EXECUTION_FAILED | GPU程序执行失败。这通常是由于 GPU 上的内核启动失败引起的，这可能由多种原因引起。                          |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_INTERNAL_ERROR   | 内部 cuBLAS 操作失败。此错误通常是由cudaMemcpyAsync()故障引起的。                                         |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_NOT_SUPPORTED    | 不支持所请求的功能。                                                                                      |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+
| CUBLAS_STATUS_LICENSE_ERROR    | 请求的功能需要一些许可证，并且在尝试检查当前许可时检测到错误。                                            |
|                                | 如果许可证不存在或已过期，或者环境变量 NVIDIA_LICENSE_FILE 设置不正确，则可能会发生此错误。               |
+--------------------------------+-----------------------------------------------------------------------------------------------------------+


cublasOperation_t
```````````````````````````````
cublasOperation_t类型指示需要对稠密矩阵执行哪种操作。

+-------------+------------+
| 值          | 说明       |
+-------------+------------+
| CUBLAS_OP_N | 非转置操作 |
+-------------+------------+
| CUBLAS_OP_T | 转置操作   |
+-------------+------------+
| CUBLAS_OP_C | 共轭转置   |
+-------------+------------+

cublasFillMode_t
```````````````````````````
指示密集矩阵的哪一部分（下部或上部）已填充,因此应由函数使用。

+------------------------+------------------+
| 值                     | 说明             |
+------------------------+------------------+
| CUBLAS_FILL_MODE_LOWER | 矩阵的下部被填充 |
+------------------------+------------------+
| CUBLAS_FILL_MODE_UPPER | 矩阵的上部被填充 |
+------------------------+------------------+
| CUBLAS_FILL_MODE_FULL  | 整个矩阵已被填充 |
+------------------------+------------------+


cublasDiagType_t
```````````````````````
该类型指示密集矩阵的主对角线是否统一，因此不应被函数触及或修改。

+----------------------+--------------------------+
| 值                   | 说明                     |
+----------------------+--------------------------+
| CUBLAS_DIAG_NON_UNIT | 矩阵对角线具有非单位元素 |
+----------------------+--------------------------+
| CUBLAS_DIAG_UNIT     | 矩阵对角线具有单位元素   |
+----------------------+--------------------------+


cublasSideMode_t
```````````````````````````````
类型指示稠密矩阵在由特定函数求解的矩阵方程中是位于左侧还是右侧。

+-------------------+--------------------+
| 值                | 说明               |
+-------------------+--------------------+
| CUBLAS_SIDE_LEFT  | 矩阵位于方程的左侧 |
+-------------------+--------------------+
| CUBLAS_SIDE_RIGHT | 矩阵位于方程的右侧 |
+-------------------+--------------------+

cublasPointerMode_t
`````````````````````````````
cublasPointerMode_t类型指示标量值是通过主机还是设备上的引用传递。
需要指出的是，如果函数调用中存在多个标量值，则它们都必须符合相同的单指针模式。
可以分别使用cublasSetPointerMode()和cublasGetPointerMode()例程设置和检索指针模式。

+----------------------------+--------------------------+
| 值                         | 说明                     |
+----------------------------+--------------------------+
| CUBLAS_POINTER_MODE_HOST   | 标量在主机上通过引用传递 |
+----------------------------+--------------------------+
| CUBLAS_POINTER_MODE_DEVICE | 标量在设备上通过引用传递 |
+----------------------------+--------------------------+

cublasAtomicsMode_t
```````````````````````````````
该类型指示是否可以使用具有使用原子的替代实现的 cuBLAS 例程。
原子模式可以分别使用cublasSetAtomicsMode()和cublasGetAtomicsMode()以及例程来设置和查询。

+----------------------------+----------------+
| 值                         | 说明           |
+----------------------------+----------------+
| CUBLAS_ATOMICS_NOT_ALLOWED | 不允许使用原子 |
+----------------------------+----------------+
| CUBLAS_ATOMICS_ALLOWED     | 允许使用原子   |
+----------------------------+----------------+

cublasGemmAlgo_t
```````````````````````````````
cublasGemmAlgo_t 类型是一个枚举，用于指定 GPU 架构上矩阵-矩阵乘法的算法，
最高可达sm_75. 在更新的 GPU 架构上sm_80，此枚举没有任何效果。


+---------------------------------------+---------------------------------------------------------------------------+
| 值                                    | 说明                                                                      |
+---------------------------------------+---------------------------------------------------------------------------+
| CUBLAS_GEMM_DEFAULT                   | 应用启发式方法选择 GEMM 算法                                              |
+---------------------------------------+---------------------------------------------------------------------------+
| CUBLAS_GEMM_ALGO0到CUBLAS_GEMM_ALGO23 | 明确选择算法 [0,23]。注意：对 NVIDIA Ampere 架构 GPU 及更新版本没有影响。 |
+---------------------------------------+---------------------------------------------------------------------------+

cublasMath_t
`````````````````````
cublasSetMathMode()中使用cublasMath_t枚举类型来选择下表中定义的计算精度模式。
由于此设置不直接控制 Tensor Core 的使用，因此该模式CUBLAS_TENSOR_OP_MATH已被弃用，并将在未来版本中删除。

+--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 值                                               | 说明                                                                                                                                                                         |
+--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_DEFAULT_MATH                              | 这是默认且性能最高的模式，使用计算和中间存储精度，且尾数和指数位数至少与请求的位数相同。只要有可能，就会使用 Tensor Core。                                                   |
+--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_PEDANTIC_MATH                             | 该模式对计算的所有阶段使用规定的精度和标准化算法，主要用于数值鲁棒性研究、测试和调试。此模式的性能可能不如其他模式。                                                         |
+--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_TF32_TENSOR_OP_MATH                       | 使用TF32 tensor core加速单精度例程                                                                                                                                           |
+--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION | 在混合精度例程中，当输出类型精度小于计算类型精度时，强制使用累加器类型（即计算类型）而不是输出类型进行矩阵乘法运算。这是一个可以与其他值一起设置的标志（使用位操作或操作）。 |
+--------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


cublasComputeType_t
`````````````````````````````
cublasComputeType_t枚举类型用于cublasGemmEx()和cublasLtMatmul()（包括所有批处理和跨步批处理变体）来选择如下定义的计算精度模式。


+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| 值                           | 说明                                                                                                                                          |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_16F           | 这是 16 位半精度浮点以及至少 16 位半精度的所有计算和中间存储精度的默认且最高性能模式。只要有可能，就会使用 Tensor Core。                      |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_16F_PEDANTIC  | 该模式对所有计算阶段使用 16 位半精度浮点标准化算法，主要用于数值鲁棒性研究、测试和调试。此模式的性能可能不如其他模式，因为它禁用tensor core。 |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32F           | 这是默认的 32 位单精度浮点，并使用至少 32 位的计算和中间存储精度。                                                                            |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32F_PEDANTIC  | 对计算的所有阶段使用 32 位单精度浮点运算，并禁用高斯复杂度降低 (3M) 等算法优化。                                                              |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32F_FAST_16F  | 允许库使用具有自动下转换功能的 Tensor Core 和 16 位半精度计算 32 位输入和输出矩阵。                                                           |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32F_FAST_16BF | 允许库使用具有自动下转换功能的 Tensor Core 和 bfloat16 计算 32 位输入和输出矩阵。                                                             |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32F_FAST_TF32 | 允许库使用 Tensor Core 和 TF32 计算 32 位输入和输出矩阵                                                                                       |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_64F           | 这是默认的 64 位双精度浮点，并使用至少 64 位的计算和中间存储精度。                                                                            |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_64F_PEDANTIC  | 对计算的所有阶段使用 64 位双精度浮点运算，并禁用高斯复杂度降低 (3M) 等算法优化。                                                              |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32I           | 这是默认的 32 位整数模式，并使用至少 32 位的计算和中间存储精度。                                                                              |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| CUBLAS_COMPUTE_32I_PEDANTIC  | 对计算的所有阶段都使用 32 位整数算术。                                                                                                        |
+------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+


cudaDataType_t
```````````````````````
该cudaDataType_t类型是一个枚举数，用于指定数据精度。当数据引用本身不携带类型时使用

+----------------+---------------------------------------------------------------+
| 值             | 说明                                                          |
+----------------+---------------------------------------------------------------+
| CUDA_R_16F     | 数据类型是16位实数半精度浮点                                  |
+----------------+---------------------------------------------------------------+
| CUDA_C_16F     | 数据类型是由两个代表复数的半精度浮点组成的 32 位结构          |
+----------------+---------------------------------------------------------------+
| CUDA_R_16BF    | 数据类型是 16 位实数 bfloat16 浮点                            |
+----------------+---------------------------------------------------------------+
| CUDA_C_16BF    | 数据类型是由两个表示复数的 bfloat16 浮点组成的 32 位结构      |
+----------------+---------------------------------------------------------------+
| CUDA_R_32F     | 数据类型是32位实数单精度浮点                                  |
+----------------+---------------------------------------------------------------+
| CUDA_C_32F     | 数据类型是 64 位结构，由两个代表复数的单精度浮点组成          |
+----------------+---------------------------------------------------------------+
| CUDA_R_64F     | 数据类型为 64 位实双精度浮点                                  |
+----------------+---------------------------------------------------------------+
| CUDA_C_64F     | 数据类型是由两个代表复数的双精度浮点组成的 128 位结构         |
+----------------+---------------------------------------------------------------+
| CUDA_R_8I      | 数据类型是 8 位实数有符号整数                                 |
+----------------+---------------------------------------------------------------+
| CUDA_C_8I      | 数据类型是一个 16 位结构，由两个代表复数的 8 位有符号整数组成 |
+----------------+---------------------------------------------------------------+
| CUDA_R_8U      | 数据类型是8位实数无符号整数                                   |
+----------------+---------------------------------------------------------------+
| CUDA_C_8U      | 数据类型是一个 16 位结构，由两个代表复数的 8 位无符号整数组成 |
+----------------+---------------------------------------------------------------+
| CUDA_R_32I     | 数据类型是 32 位实数有符号整数                                |
+----------------+---------------------------------------------------------------+
| CUDA_C_32I     | 数据类型是 64 位结构，由两个表示复数的 32 位有符号整数组成    |
+----------------+---------------------------------------------------------------+
| CUDA_R_8F_E4M3 | 数据类型为E4M3格式的8位实浮点数                               |
+----------------+---------------------------------------------------------------+
| CUDA_R_8F_E5M2 | 数据类型为E5M2格式的8位实浮点数                               |
+----------------+---------------------------------------------------------------+


libraryPropertyType_t
`````````````````````````````````
libraryPropertyType_t用作参数来指定在使用cublasGetProperty()时请求哪个属性

+---------------+------------------------+
| 值            | 说明                   |
+---------------+------------------------+
| MAJOR_VERSION | 查询主要版本的枚举     |
+---------------+------------------------+
| MINOR_VERSION | 查询次要版本的枚举     |
+---------------+------------------------+
| PATCH_LEVEL   | 用于标识补丁级别的编号 |
+---------------+------------------------+


cublas辅助函数
---------------------------

cublasCreate
`````````````````````````
该函数初始化 cuBLAS 库并创建一个保存 cuBLAS 库上下文的不透明结构的句柄。
它在主机和设备上分配硬件资源，并且必须在进行任何其他 cuBLAS 库调用之前调用。
cuBLAS 库上下文与当前 CUDA 设备绑定。要在多个设备上使用该库，需要为每个设备创建一个 cuBLAS 句柄。
此外，对于给定设备，可以创建具有不同配置的多个 cuBLAS 句柄。

因为cublasCreate()分配了一些内部资源，并且通过调用cublasDestroy()释放这些资源将隐式调用cudaDeviceSynchronize()，
建议尽量减少这些函数的调用次数。对于从不同线程使用同一设备的多线程应用程序，
建议的编程模型是为每个线程创建一个 cuBLAS 句柄，并在线程的整个生命周期中使用该 cuBLAS 句柄。

.. code-block:: cpp

    cublasStatus_t cublasCreate(cublasHandle_t *handle);

cublasDestroy
```````````````````
该函数释放cuBLAS库使用的硬件资源。该函数通常是最后一次调用cuBLAS库的特定句柄。
由于cublasCreate()函数分配了一些内部资源，
调用cublasDestroy()函数释放这些资源将隐式调用cudaDeviceSynchronize()函数，因此建议尽量减少调用这些函数的次数。

.. code-block:: cpp

    cublasStatus_t cublasDestroy(cublasHandle_t handle);

cublasGetVersion
```````````````````````
此函数返回 cuBLAS 库的版本号。

.. code-block:: cpp

    cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version);

cublasGetProperty
```````````````````````````
该函数返回 value 所指向的内存中所请求属性的值

.. code-block:: cpp

    cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value);

cublasGetStatusName
`````````````````````````````
该函数返回给定状态的字符串表示形式。

.. code-block:: cpp

    const char* cublasGetStatusName(cublasStatus_t status);

cublasGetStatusString
```````````````````````````````
此函数返回给定状态的描述字符串。

.. code-block:: cpp

    const char* cublasGetStatusString(cublasStatus_t status);

cublasSetStream
```````````````````````
该函数设置cuBLAS库流，它将用于执行所有后续的cuBLAS库函数调用。
如果没有设置cuBLAS库流，所有内核都将使用默认流。
特别的，该例程可用于在内核启动之间更改流，然后将cuBLAS库流重置为NULL。
此外，该函数无条件地将cuBLAS库工作区重置为默认工作区池（参见cublasSetWorkspace()）。

.. code-block:: cpp

    cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);

cublasSetWorkspace
```````````````````````````
此函数将 cuBLAS 库工作区设置为用户拥有的设备缓冲区，该缓冲区将用于执行对 cuBLAS 库函数的所有后续调用（在当前设置的流上）。
如果未设置 cuBLAS 库工作空间，则所有内核将使用在 cuBLAS 上下文创建期间分配的默认工作空间池。

特别是，该例程可用于更改内核启动之间的工作空间。工作区指针必须对齐到至少 256 字节，否则CUBLAS_STATUS_INVALID_VALUE将返回错误。
cublasSetStream ()函数无条件地将 cuBLAS 库工作区重置回默认工作区池。
太小workspaceSizeInBytes可能会导致某些例程失败CUBLAS_STATUS_ALLOC_FAILED返回错误或导致性能大幅下降。
等于或大于 16KiB 的工作空间大小足以防止CUBLAS_STATUS_ALLOC_FAILED错误，而更大的工作空间可以为某些例程提供性能优势。

+--------------------+--------------------+
| GPU架构            | 推荐的工作空间尺寸 |
+--------------------+--------------------+
| NVIDIA Hopper 架构 | 32 兆字节          |
+--------------------+--------------------+
| 其他               | 4MB                |
+--------------------+--------------------+

.. code-block:: cpp

    cublasStatus_t 
    cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes);

cublasGetStream
```````````````````````
该函数获取 cuBLAS 库流，该流用于执行对 cuBLAS 库函数的所有调用。如果未设置 cuBLAS 库流，则所有内核都使用默认流

.. code-block:: cpp

    cublasStatus_t
    cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId);

cublasGetPointerMode
```````````````````````````````
该函数获取cuBLAS库使用的指针模式。

.. code-block:: cpp

    cublasStatus_t
    cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode);

cublasSetPointerMode
`````````````````````````````
该函数设置cuBLAS库使用的指针模式。

.. code-block:: cpp

    cublasStatus_t
    cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);

cublasSetVector
`````````````````````````
该函数支持64位int接口

此函数将n个元素从 GPU 内存空间中的x向量复制到主机内存空间中的y向量。

由于假定二维矩阵采用列主格式，因此如果向量是矩阵的一部分，则向量增量等于1访问该矩阵的（部分）列。
类似地，使用等于矩阵主维的增量会导致访问该矩阵的（部分）行。

.. code-block:: cpp

    cublasStatus_t
    cublasGetVector(int n, int elemSize,
                const void *x, int incx, void *y, int incy);
    // n：元素个数
    // elemSize:元素的字节大小
    // x:源地址
    // incx:元素间存储间距
    // y:目标地址
    // incy:元素间存储间距

cublasSetMatrix
`````````````````````````
该函数支持64 位整数接口。

此函数将元素图块从主机内存空间中的矩阵复制到GPU 内存空间中的矩阵。

.. code-block:: cpp

    cublasStatus_t
    cublasSetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb);
    // rows：矩阵的行数。
    // cols：矩阵的列数。
    // elemSize：每个元素的大小，以字节为单位。
    // A：指向主机内存中的矩阵数据的指针。
    // lda：矩阵的 leading dimension，即每一行的字节数。
    // B：指向设备内存中的矩阵数据的指针。
    // ldb：矩阵的 leading dimension，即每一行的字节数。

cublasGetMatrix
```````````````````````
.. code-block:: cpp

    cublasStatus_t
    cublasGetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb);

cublasSetVectorAsync
`````````````````````````````
此函数与cublasSetVector()具有相同的功能，不同之处在于数据传输是使用给定的 CUDA™ 流参数异步完成的

.. code-block:: cpp

    cublasStatus_t
    cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx,
                     void *devicePtr, int incy, cudaStream_t stream);

cublasGetVectorAsync
```````````````````````````````
.. code-block:: cpp

    cublasStatus_t
    cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx,
                     void *hostPtr, int incy, cudaStream_t stream);

cublasSetMatrixAsync
`````````````````````````````
.. code-block:: cpp

    cublasStatus_t
    cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                     int lda, void *B, int ldb, cudaStream_t stream);

cublasGetMatrixAsync
```````````````````````````````
.. code-block:: cpp

    cublasStatus_t
    cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                     int lda, void *B, int ldb, cudaStream_t stream);

cublasSetAtomicsMode
```````````````````````````
一些例程（例如cublas<t>symv和cublas<t>hemv）具有使用原子来累积结果的替代实现。
这种实现通常要快得多，但每次运行生成的结果可能与其他运行不完全相同。
从数学上讲，这些不同的结果并不重要，但在调试时这些差异可能会产生不利影响。

此函数允许或禁止在 cuBLAS 库中对具有备用实现的所有例程使用原子。
如果没有在任何 cuBLAS 例程的文档中明确指定，则意味着该例程没有使用原子的替代实现。
当原子模式被禁用时，当在同一硬件上使用相同的参数调用时，每个 cuBLAS 例程应该从一次运行到另一次运行产生相同的结果

默认初始化的cublasHandle_t对象的默认原子模式是CUBLAS_ATOMICS_NOT_ALLOWED

.. code-block:: cpp

    cublasStatus_t cublasSetAtomicsMode(cublasHandlet handle, cublasAtomicsMode_t mode);

cublasGetAtomicsMode
```````````````````````````
.. code-block:: cpp

    cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode);

cublasSetMathMode
```````````````````````````````
cublasSetMathMode ()函数使您能够选择cublasMath_t定义的计算精度模式。用户可以将计算精度模式设置为它们的逻辑组合

cublasGetMathMode
```````````````````````````
.. code-block:: cpp

    cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode);

cublasSetSmCountTarget
```````````````````````````````````
cublasSetSmCountTarget ()函数允许在内核执行期间覆盖库可用的多处理器数量。

当设置为0库时，将返回其默认行为。输入值不应超过设备的多处理器数量，可以使用 获取该数量cudaDeviceGetAttribute。不接受负值。

用户在使用此例程修改库句柄时必须确保线程安全，类似于使用cublasSetStream()等。

.. code-block:: cpp

    cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget)


cublasGetSmCountTarget
`````````````````````````````````````
该函数获取先前编程到库句柄的值。

.. code-block:: cpp

    cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget)

cublasLoggerConfigure
```````````````````````````````````
该函数在运行时配置日志记录。除了这种类型的配置之外，还可以使用特殊的环境变量来配置日志记录，这些变量将由 libcublas 检查：

* CUBLAS_LOGINFO_DBG - 设置环境变量设置为“1”表示打开日志记录（默认情况下日志记录关闭）。
* CUBLAS_LOGDEST_DBG - 设置环境变量编码如何记录。“stdout”、“stderr”分别表示将日志消息输出到stdout或stderr。
  在另一种情况下，它指定文件的“文件名”。

.. code-block:: cpp

    cublasStatus_t cublasLoggerConfigure(
    int             logIsOn,
    int             logToStdOut,
    int             logToStdErr,
    const char*     logFileName);
    // logIsOn:完全打开/关闭日志记录。默认情况下是关闭的，但可以通过调用cublasSetLoggerCallback()用户定义的回调函数来打开。
    // logToStdOut:打开/关闭对标准输出 I/O 流的记录。默认情况下是关闭的
    // logToStdErr:打开/关闭对标准错误 I/O 流的记录。默认情况下是关闭的。
    // logFileName:打开/关闭对文件系统中由其名称指定的文件的日志记录。
    //        cublasLoggerConfigure()复制logFileName. 如果您对这种类型的日志记录不感兴趣，您应该提供空指针。

cublasGetLoggerCallback
``````````````````````````````````
此函数通过cublasSetLoggerCallback()检索指向先前安装的自定义用户定义回调函数的函数指针，否则为零。

.. code-block:: cpp

    cublasStatus_t cublasGetLoggerCallback(
        cublasLogCallback* userCallback);
    // userCallback:指向用户定义的回调函数的指针

cublasSetLoggerCallback
```````````````````````````````````
该函数通过cublas C公共API安装用户自定义的回调函数。

.. code-block:: cpp

    cublasStatus_t cublasSetLoggerCallback(
        cublasLogCallback   userCallback);

cuBLAS 1 级函数
----------------------------
基于标量和向量的运算的 1 级基本线性代数子程序 (BLAS1) 函数

使用缩写 <type>表示类型，使用缩写 <t>表示相应的短类型，以便更简洁、清晰地表达所实现的功能。

+-----------------+------------+------------+
| <type>          | <t>        | 说明       |
+-----------------+------------+------------+
| float           | 's' 或 'S' | 实数单精度 |
+-----------------+------------+------------+
| double          | “d”或“D”   | 实数双精度 |
+-----------------+------------+------------+
| cuComplex       | “c”或“C”   | 复杂单精度 |
+-----------------+------------+------------+
| cuDoubleComplex | “z”或“Z”   | 复杂双精度 |
+-----------------+------------+------------+

缩写 **Re(.)** 和 **Im(.)** 分别代表数字的实部和虚部。
由于实数的虚部不存在，因此我们将其视为零，并且通常可以简单地从使用它的方程中将其丢弃。
另外，:math:`\bar{\alpha}` 表示复共轭

cublasI<t>amax()
`````````````````````````````
支持64位整数

此函数查找最大值元素的（最小）索引。

.. code-block:: cpp

    cublasStatus_t cublasIsamax(cublasHandle_t handle, int n,
                            const float *x, int incx, int *result);
    cublasStatus_t cublasIdamax(cublasHandle_t handle, int n,
                                const double *x, int incx, int *result);
    cublasStatus_t cublasIcamax(cublasHandle_t handle, int n,
                                const cuComplex *x, int incx, int *result);
    cublasStatus_t cublasIzamax(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, int *result);
    // n:元素个数
    // x:设备内存中的数据指针
    // incx:连续元素之间的步幅，以字节为单位
    // result:结果，可以在主机内存或设备内存

cublasI<t>amin()
```````````````````````````
支持64位整数

该函数查找最小元素的（最小）索引。

.. code-block:: cpp

    cublasStatus_t cublasIsamin(cublasHandle_t handle, int n,
                            const float *x, int incx, int *result);
    cublasStatus_t cublasIdamin(cublasHandle_t handle, int n,
                                const double *x, int incx, int *result);
    cublasStatus_t cublasIcamin(cublasHandle_t handle, int n,
                                const cuComplex *x, int incx, int *result);
    cublasStatus_t cublasIzamin(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, int *result);

cublas<t>asum()
`````````````````````````
支持64位整数

该函数计算向量x的绝对值之和。

.. code-block:: cpp

    cublasStatus_t  cublasSasum(cublasHandle_t handle, int n,
                            const float *x, int incx, float  *result);
    cublasStatus_t  cublasDasum(cublasHandle_t handle, int n,
                                const double *x, int incx, double *result);
    cublasStatus_t cublasScasum(cublasHandle_t handle, int n,
                                const cuComplex *x, int incx, float  *result);
    cublasStatus_t cublasDzasum(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, double *result);


cublas<t>axpy()
```````````````````````````
支持64位整数

该函数将向量x乘以标量 :math:`\alpha`,再和向量y相加，结果写入到向量y中。

.. code-block:: cpp

    cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy);
    cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n,
                            const double          *alpha,
                            const double          *x, int incx,
                            double                *y, int incy);
    cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n,
                            const cuComplex       *alpha,
                            const cuComplex       *x, int incx,
                            cuComplex             *y, int incy);
    cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *x, int incx,
                            cuDoubleComplex       *y, int incy);
    // alpha:用于乘法的标量，可以在主机或设备内存中
    // n:向量x，y中元素个数
    // x:向量x的起始地址，向量x必须在设备内存中
    // incx:向量x连续元素之间的步幅，以字节为单位
    // y:向量y的起始地址，向量y必须在设备内存中
    // incy:向量y连续元素之间的步幅，以字节为单位

cublas<t>copy()
```````````````````````````
该函数将向量x复制到向量y。

.. code-block:: cpp

    cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           float                 *y, int incy);
    cublasStatus_t cublasDcopy(cublasHandle_t handle, int n,
                            const double          *x, int incx,
                            double                *y, int incy);
    cublasStatus_t cublasCcopy(cublasHandle_t handle, int n,
                            const cuComplex       *x, int incx,
                            cuComplex             *y, int incy);
    cublasStatus_t cublasZcopy(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx,
                            cuDoubleComplex       *y, int incy);
    // x和y必须在设备内存中

cublas<t>dot()
`````````````````````````
该函数支持64 位整数接口。

该函数计算向量x和y的点积

.. code-block:: cpp

    cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result);
    cublasStatus_t cublasDdot (cublasHandle_t handle, int n,
                            const double          *x, int incx,
                            const double          *y, int incy,
                            double          *result);
    cublasStatus_t cublasCdotu(cublasHandle_t handle, int n,
                            const cuComplex       *x, int incx,
                            const cuComplex       *y, int incy,
                            cuComplex       *result);
    cublasStatus_t cublasCdotc(cublasHandle_t handle, int n,
                            const cuComplex       *x, int incx,
                            const cuComplex       *y, int incy,
                            cuComplex       *result);
    cublasStatus_t cublasZdotu(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx,
                            const cuDoubleComplex *y, int incy,
                            cuDoubleComplex *result);
    cublasStatus_t cublasZdotc(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx,
                            const cuDoubleComplex *y, int incy,
                            cuDoubleComplex       *result);
    // n:向量x和y中的元素数量
    // x,y必须在设备内存中
    // result:得到的点积，0.0 if n<=0。可在设备后主机内存中

cublas<t>nrm2()
`````````````````````````
该函数计算向量x的欧几里德范数(平方和在开方)

.. code-block:: cpp

    cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result);
    cublasStatus_t  cublasDnrm2(cublasHandle_t handle, int n,
                                const double          *x, int incx, double *result);
    cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n,
                                const cuComplex       *x, int incx, float  *result);
    cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n,
                                const cuDoubleComplex *x, int incx, double *result);
    // result:由此产生的范数，0.0 if n,incx<=0。,可以在设备或主机内存中

cublas<t>scal()
```````````````````````
该函数支持64 位整数接口。

该函数按标量 :math:`\alpha` 缩放向量x并用结果覆盖它。

.. code-block:: cpp

    cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx)
    cublasStatus_t  cublasDscal(cublasHandle_t handle, int n,
                                const double          *alpha,
                                double          *x, int incx)
    cublasStatus_t  cublasCscal(cublasHandle_t handle, int n,
                                const cuComplex       *alpha,
                                cuComplex       *x, int incx)
    cublasStatus_t cublasCsscal(cublasHandle_t handle, int n,
                                const float           *alpha,
                                cuComplex       *x, int incx)
    cublasStatus_t  cublasZscal(cublasHandle_t handle, int n,
                                const cuDoubleComplex *alpha,
                                cuDoubleComplex *x, int incx)
    cublasStatus_t cublasZdscal(cublasHandle_t handle, int n,
                                const double          *alpha,
                                cuDoubleComplex *x, int incx)
    // alpha:用于乘法的标量,可以在设备会主机内存中

cublas<t>swap()
```````````````````````````
该函数支持64 位整数接口。

该函数交换向量x和y的元素。

.. code-block:: cpp

    cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float           *x,
                           int incx, float           *y, int incy)
    cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double          *x,
                            int incx, double          *y, int incy)
    cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex       *x,
                            int incx, cuComplex       *y, int incy)
    cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex *x,
                            int incx, cuDoubleComplex *y, int incy)

cuBLAS 2 级函数
------------------------------
执行矩阵向量运算的 2 级基本线性代数子程序 (BLAS2) 函数。

cublas<t>gbmv()
```````````````````````````
该函数支持64 位整数接口。

该函数执行带状矩阵向量乘法

.. code-block:: cpp

    cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, int kl, int ku,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)
    cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, int kl, int ku,
                            const double          *alpha,
                            const double          *A, int lda,
                            const double          *x, int incx,
                            const double          *beta,
                            double          *y, int incy)
    cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, int kl, int ku,
                            const cuComplex       *alpha,
                            const cuComplex       *A, int lda,
                            const cuComplex       *x, int incx,
                            const cuComplex       *beta,
                            cuComplex       *y, int incy)
    cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, int kl, int ku,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *x, int incx,
                            const cuDoubleComplex *beta,
                            cuDoubleComplex *y, int incy)
    // trans:控制转置的参数
    // m:矩阵A的行数
    // n:矩阵A的列数
    // kl:矩阵A的下对角线数
    // ku:矩阵A的上对角线数
    // alpha:用于乘法的标量
    // A:矩阵在设备上的地址
    // lda:用于存储矩阵A的二维数组的主维
    // x:向量在设备上的地址
    // incx:x中连续元素的步长
    // beta:用于乘法的标量
    // y:输出结果，位于设备上
    // incy:y中连续元素的步长

cuBLAS 3 级函数
-------------------------
执行矩阵-矩阵运算的 3 级基本线性代数子程序 (BLAS3) 函数

cublas<t>gemm()
`````````````````````````````````
该函数支持64 位整数接口。

该函数执行矩阵-矩阵乘法

C= :math:`\alpha` AB + :math:`\beta` C

.. code-block:: cpp

    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
    cublasStatus_t cublasDgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const double          *alpha,
                            const double          *A, int lda,
                            const double          *B, int ldb,
                            const double          *beta,
                            double          *C, int ldc)
    cublasStatus_t cublasCgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const cuComplex       *alpha,
                            const cuComplex       *A, int lda,
                            const cuComplex       *B, int ldb,
                            const cuComplex       *beta,
                            cuComplex       *C, int ldc)
    cublasStatus_t cublasZgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const cuDoubleComplex *beta,
                            cuDoubleComplex *C, int ldc)
    cublasStatus_t cublasHgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const __half *alpha,
                            const __half *A, int lda,
                            const __half *B, int ldb,
                            const __half *beta,
                            __half *C, int ldc)
    // A:mxk B:kxn C:mxn
    // transa:矩阵A的转置类型，cublas是列优先存储的，要使用行优先需要使用CUBLAS_OP_T，并修改矩阵的宽高
    // transb:矩阵B的转置类型
    // m:矩阵op(A)和C的行数
    // n:矩阵op(B)和C的列数
    // k:矩阵op(A)的列数，矩阵op(B)的行数
    // alpha:用于乘法的标量
    // A:矩阵A的数据，在设备上，如果transa == CUBLAS_OP_N，维度为lda x k(要求lda>=max(1,m))
    //   否则维度为lda x m (要求lda>=max(1,k))
    // lda:矩阵A的主维。
    // B:矩阵B的数据，在设备上，如果transa == CUBLAS_OP_N，维度为ldb x n(要求ldb>=max(1,k))
    //   否则，维度为ldb x k (要求ldb>=max(1,n))
    // ldb:矩阵B的主维。
    // beta:用于乘法的标量
    // C:矩阵C的数据，在设备上,维度为ldc x n (要求ldc>=max(1,m))
    //   输出是列优先布局
    // ldc:矩阵C的主维。

