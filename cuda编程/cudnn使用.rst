cudnn使用
=====================
api: https://docs.nvidia.com/deeplearning/cudnn/api/index.html

cudnn_ops_infer.so库
------------------------------------

指向Opaque结构体的指针
`````````````````````````````

cudnnActivationDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnActivationDescriptor_t是一个指向不透明结构的指针，该结构保存激活操作的描述。
cudnnCreateActivationDescriptor()用于创建一个实例，并且cudnnSetActivationDescriptor()必须用于初始化该实例。

cudnnCTCLossDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnCTCLossDescriptor_t是一个指向不透明结构的指针，该结构保存了 CTC 丢失操作的描述。
cudnnCreateCTCLossDescriptor()用于创建一个实例，cudnnSetCTCLossDescriptor()用于初始化该实例，
cudnnDestroyCTCLossDescriptor()用于销毁该实例。

cudnnDropoutDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnDropoutDescriptor_t是一个指向不透明结构的指针，该结构保存了 dropout 操作的描述。
cudnnCreateDropoutDescriptor()用于创建一个实例，cudnnSetDropoutDescriptor()用于初始化该实例，
cudnnDestroyDropoutDescriptor()用于销毁该实例，
cudnnGetDropoutDescriptor()用于查询先前初始化实例的字段，
cudnnRestoreDropoutDescriptor()用于将实例恢复到先前保存的关闭状态。

cudnnFilterDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnFilterDescriptor_t是指向保存卷积的kenerl描述的不透明结构的指针。
cudnnCreateFilterDescriptor()用于创建一个实例，
并且必须使用cudnnSetFilter4dDescriptor()或cudnnSetFilterNdDescriptor()来初始化该实例。

cudnnHandle_t
~~~~~~~~~~~~~~~~~~~~~~~
cudnnHandle_t是指向保存 cuDNN 库上下文的不透明结构的指针。
必须使用cudnnCreate()创建cuDNN 库上下文，并且返回的句柄必须传递给所有后续库函数调用。
最后应使用 销毁上下文cudnnDestroy()。
该上下文仅与一个 GPU 设备关联，即调用cudnnCreate()时的当前设备。但是，可以在同一 GPU 设备上创建多个上下文。

对于cuDNN的cudnnHandle_t对象，应该避免在多线程之间共享同一个实例。
这是因为cudnnHandle_t对象并不是线程安全的，多线程之间共享同一个cudnnHandle_t会导致潜在的竞争条件和不确定的行为。

为了在多线程环境中使用cuDNN，每个线程应该拥有自己独立的cudnnHandle_t对象。
这样可以避免线程之间的干扰，确保cuDNN操作的正确性和稳定性。

cudnnLRNDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnLRNDescriptor_t是指向保存本地响应标准化参数的不透明结构的指针。
cudnnCreateLRNDescriptor()用于创建一个实例，并且必须使用cudnnSetLRNDescriptor()来初始化该实例。

cudnnOpTensorDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnOpTensorDescriptor_t是一个指向不透明结构的指针，该结构保存 Tensor Core 操作的描述，用作cudnnOpTensor()的参数。
cudnnCreateOpTensorDescriptor()用于创建一个实例，并且cudnnSetOpTensorDescriptor()必须用于初始化该实例。

cudnnPoolingDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnPoolingDescriptor_t是一个指向池化操作描述的不透明结构的指针。
cudnnCreatePoolingDescriptor()用于创建一个实例，
并且必须使用cudnnSetPoolingNdDescriptor()或cudnnSetPooling2dDescriptor()来初始化该实例。

cudnnReduceTensorDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnReduceTensorDescriptor_t是一个指向不透明结构的指针，该结构保存张量归约操作的描述，用作cudnnReduceTensor()的参数。
cudnnCreateReduceTensorDescriptor()用于创建一个实例，并且cudnnSetReduceTensorDescriptor()必须用于初始化该实例。

cudnnSpatialTransformerDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnSpatialTransformerDescriptor_t是一个指向不透明结构的指针，该结构保存空间变换操作的描述。
cudnnCreateSpatialTransformerDescriptor()用于创建一个实例，
cudnnSetSpatialTransformerNdDescriptor()用于初始化该实例，cudnnDestroySpatialTransformerDescriptor()用于销毁该实例。

cudnnTensorDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnTensorDescriptor_t是一个指向不透明结构的指针，该结构保存通用 nD 数据集的描述。
cudnnCreateTensorDescriptor()用于创建一个实例，
并且必须使用cudnnSetTensorNdDescriptor(),cudnnSetTensor4dDescriptor(),cudnnSetTensor4dDescriptorEx()
其中一个例程或来初始化该实例。 

cudnnTensorTransformDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnTensorTransformDescriptor_t是一个不透明的结构，包含张量变换的描述。
使用该cudnnCreateTensorTransformDescriptor()函数创建该描述符的实例，
并使用cudnnDestroyTensorTransformDescriptor()该函数销毁先前创建的实例。

枚举类型
``````````````

cudnnActivationMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnActivationMode_t是一个枚举类型，
用于选择cudnnActivationForward()、cudnnActivationBackward()和cudnnConvolutionBiasActivationForward()中使用的神经元激活函数。


* CUDNN_ACTIVATION_SIGMOID:选择 sigmoid 函数 
* CUDNN_ACTIVATION_RELU:relu激活函数      
* CUDNN_ACTIVATION_TANH:tanh激活函数      
* CUDNN_ACTIVATION_CLIPPED_RELU:clip relu激活
* CUDNN_ACTIVATION_ELU:指数线性函数      
* CUDNN_ACTIVATION_IDENTITY:不激活，直接穿透  


cudnnBatchNormMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnBatchNormOps_t是一个枚举类型，用于指定
cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize()、cudnnBatchNormalizationForwardTrainingEx()、
cudnnGetBatchNormalizationBackwardExWorkspaceSize()、cudnnBatchNormalizationBackwardEx()和
cudnnGetBatchNormalizationTrainingExReserveSpaceSize()函数中的操作模式。

* CUDNN_BATCHNORM_OPS_BN:每次激活时仅执行批量归一化
* CUDNN_BATCHNORM_OPS_BN_ACTIVATION:首先进行批量归一化，然后进行激活。
* CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION:执行批量归一化，然后按元素加法，然后执行激活操作

cudnnCTCLossAlgo_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnCTCLossAlgo_t是一个枚举类型，公开了可用于执行 CTC 损失操作的不同算法。

* CUDNN_CTC_LOSS_ALGO_DETERMINISTIC:结果保证是可重复的
* CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC:不保证结果可重复。

cudnnDataType_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnDataType_t是一个枚举类型，指示张量描述符或滤波器描述符引用的数据类型。

* CUDNN_DATA_FLOAT:数据是 32 位单精度浮点 ( float)
* CUDNN_DATA_DOUBLE:数据是 64 位双精度浮点 ( double)
* CUDNN_DATA_HALF:数据是 16 位浮点数。
* CUDNN_DATA_INT8:数据是一个 8 位有符号整数。
* CUDNN_DATA_INT32:数据是一个 32 位有符号整数。
* CUDNN_DATA_INT8x4:数据是 32 位元素，每个元素由 4 个 8 位有符号整数组成。此数据类型仅支持张量格式CUDNN_TENSOR_NCHW_VECT_C。
* CUDNN_DATA_UINT8:数据是一个 8 位无符号整数。
* CUDNN_DATA_UINT8x4:数据是 32 位元素，每个元素由 4 个 8 位无符号整数组成。此数据类型仅支持张量格式CUDNN_TENSOR_NCHW_VECT_C。
* CUDNN_DATA_INT8x32:数据是 32 元素向量，每个元素都是 8 位有符号整数。此数据类型仅支持张量格式CUDNN_TENSOR_NCHW_VECT_C。
  此外，该数据类型只能与algo 1, 意思是一起使用CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM。有关详细信息，请参阅cudnnConvolutionFwdAlgo_t。
* CUDNN_DATA_BFLOAT16：数据为 16 位量，其中 7 位尾数位、8 位指数位和 1 位符号位。
* CUDNN_DATA_INT64：数据是 64 位有符号整数。
* CUDNN_DATA_BOOLEAN：数据是布尔值 ( bool)。
  请注意，对于 type CUDNN_TYPE_BOOLEAN，元素应被“打包”：即一个字节包含 8 个 type 元素CUDNN_TYPE_BOOLEAN。
  此外，在每个字节内，元素从最低有效位到最高有效位进行索引。
  例如，包含 01001111 的 8 个元素的 1 维张量对于元素 0 到 3 具有值 1，对于元素 4 和 5 具有值 0，对于元素 6 具有值 1，对于元素 7 具有值 0。
  超过 8 个元素的张量仅使用更多字节，其中顺序也是从最低有效字节到最高有效字节。
  请注意，CUDA 是小端字节序，这意味着最低有效字节具有较低的内存地址。
  例如，在 16 个元素的情况下，01001111 11111100 对于元素 0 到 3 具有值 1，对于元素 4 和 5 具有值 0，对于元素 6 具有值 1，对于元素 7 具有值 0，对于元素 8 和 9 具有值 0，对于元素 10 具有值 1到 15。
* CUDNN_DATA_FP8_E4M3：数据为 8 位量，其中 3 位尾数位、4 位指数位和 1 位符号位。
* CUDNN_DATA_FP8_E5M2：The data is an 8-bit quantity, with 2 mantissa bits, 5 exponent bits, and 1 sign bit.
* CUDNN_DATA_FAST_FLOAT_FOR_FP8：CUDNN_DATA_FLOAT该数据类型是用于 FP8 张量核心运算的 吞吐量较高但精度较低的计算类型（与 相比）

cudnnDeterminism_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnDeterminism_t是一个枚举类型，用于指示计算结果是否是确定性的（可重现的）

* CUDNN_NON_DETERMINISTIC：不保证结果可重复。
* CUDNN_DETERMINISTIC：结果保证是可重复的

cudnnDivNormMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnDivNormMode_t是一个枚举类型，
用于指定cudnnDivisiveNormalizationForward()和cudnnDivisiveNormalizationBackward()中的操作模式。

* CUDNN_DIVNORM_PRECOMPUTED_MEANS：均值张量数据指针应包含用户预先计算的均值或其他内核卷积值。均值指针也可以是NULL，在这种情况下，它被认为是用零填充的。这相当于空间LRN

cudnnErrQueryMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnErrQueryMode_t是传递给cudnnQueryRuntimeError()选择远程内核错误查询模式的枚举类型。

* CUDNN_ERRQUERY_RAWCODE:无论内核完成状态如何，都读取错误存储位置
* CUDNN_ERRQUERY_NONBLOCKING:报告 cuDNN 句柄的用户流中的所有任务是否已完成。如果是这种情况，请报告远程内核错误代码。
* CUDNN_ERRQUERY_BLOCKING:等待用户流中的所有任务完成，然后再报告远程内核错误代码

cudnnFoldingDirection_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnFoldingDirection_t是一个枚举类型，用于选择折叠方向。

* CUDNN_TRANSFORM_FOLD = 0U : 选择折叠
* CUDNN_TRANSFORM_UNFOLD = 1U : 选择展开

cudnnIndicesType_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnIndicesType_t是一个枚举类型，用于指示cudnnReduceTensor()例程要计算的索引的数据类型。
该枚举类型用作cudnnReduceTensorDescriptor_t描述符的字段。

* CUDNN_32BIT_INDICES:计算 unsigned int 索引。
* CUDNN_64BIT_INDICES:计算无符号长索引。
* CUDNN_16BIT_INDICES:计算无符号短索引。
* CUDNN_8BIT_INDICES:计算无符号字符索引。

cudnnLRNMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnLRNMode_t是一个枚举类型，用于指定cudnnLRNCrossChannelForward()和cudnnLRNCrossChannelBackward()中的操作模式。

* CUDNN_LRN_CROSS_CHANNEL_DIM1:LRN 计算是在张量的维度dimA[1]上执行的。

cudnnMathType_t
~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnMathType_t是一个枚举类型，用于指示在给定的库例程中是否允许使用 Tensor Core 操作。

* CUDNN_DEFAULT_MATH:Tensor Core 运算不用于 NVIDIA A100 之前的 GPU 设备。在A100 GPU架构设备上，允许Tensor Core TF32操作。
* CUDNN_TENSOR_OP_MATH:允许使用 Tensor Core 操作，但不会主动对张量执行数据类型下转换以利用 Tensor Core。
* CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:允许使用 Tensor Core 操作，并将主动对张量执行数据类型下转换，以便利用 Tensor Core。
* CUDNN_FMA_MATH:仅限于使用 FMA 指令的内核。

在 NVIDIA A100 之前的 GPU 设备上，CUDNN_DEFAULT_MATH具有CUDNN_FMA_MATH相同的行为：不会选择 Tensor Core 内核。
借助 NVIDIA Ampere 架构和 CUDA 工具包 11，CUDNN_DEFAULT_MATH允许 TF32 Tensor Core 操作，但CUDNN_FMA_MATH不允许。
和其他 Tensor Core 数学类型的TF32 行为CUDNN_DEFAULT_MATH可以通过环境变量显式禁用NVIDIA_TF32_OVERRIDE=0。

cudnnNanPropagation_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnNanPropagation_t是一个枚举类型，用于指示给定例程是否应该传播Nan数字。
该枚举类型用作cudnnActivationDescriptor_t描述符和cudnnPoolingDescriptor_t描述符的字段。

* CUDNN_NOT_PROPAGATE_NAN:Nan数字不会传播。
* CUDNN_PROPAGATE_NAN:Nan数字被传播。

cudnnNormAlgo_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnNormAlgo_t是一个枚举类型，用于指定执行归一化操作的算法。

* CUDNN_NORM_ALGO_STANDARD:执行标准标准化。
* CUDNN_NORM_ALGO_PERSIST:此模式与 类似CUDNN_NORM_ALGO_STANDARD，但它仅支持CUDNN_NORM_PER_CHANNEL某些任务并且速度更快。

cudnnNormMode_t
~~~~~~~~~~~~~~~~~~~~~~~~
cudnnNormMode_t是一个枚举类型，用于指定
cudnnNormalizationForwardInference()、cudnnNormalizationForwardTraining()、
cudnnBatchNormalizationBackward()、cudnnGetNormalizationForwardTrainingWorkspaceSize()、
cudnnGetNormalizationBackwardWorkspaceSize()和cudnnGetNormalizationTrainingReserveSpaceSize()例程中的操作模式

* CUDNN_NORM_PER_ACTIVATION:标准化是在每次激活时执行的。该模式旨在在非卷积网络层之后使用。
  在此模式下，normBias和normScale的张量维度以及函数cudnnNormalization中使用的参数为 1xCxHxW。
* CUDNN_NORM_PER_CHANNEL:标准化是在 N+ 空间维度上按通道执行的。
  此模式旨在用于卷积层之后（需要空间不变性的情况）。在此模式下，normBias和normScale张量尺寸为 1xCx1x1。

cudnnNormOps_t
~~~~~~~~~~~~~~~~~~~~~~~~
cudnnNormOps_t是一个枚举类型，用于指定
cudnnGetNormalizationForwardTrainingWorkspaceSize()、cudnnNormalizationForwardTraining()、
cudnnGetNormalizationBackwardWorkspaceSize()、cudnnNormalizationBackward()
和cudnnGetNormalizationTrainingReserveSpaceSize()函数中的操作模式。

* CUDNN_NORM_OPS_NORM:仅执行标准化。
* CUDNN_NORM_OPS_NORM_ACTIVATION:首先，进行归一化，然后进行激活。
* CUDNN_NORM_OPS_NORM_ADD_ACTIVATION:执行标准化，然后按元素加法，然后执行激活操作。

cudnnOpTensorOp_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnOpTensorOp_t是一个枚举类型，用于指示cudnnOpTensor()例程要使用的 Tensor Core 操作。
该枚举类型用作描述符的字段cudnnOpTensorDescriptor_t。

* CUDNN_OP_TENSOR_ADD:要执行的操作是加法运算。
* CUDNN_OP_TENSOR_MUL:要执行的运算是乘法。
* CUDNN_OP_TENSOR_MIN:要执行的操作是最小比较。
* CUDNN_OP_TENSOR_MAX:要执行的操作是最大比较。
* CUDNN_OP_TENSOR_SQRT:要执行的运算是平方根，仅对A张量执行。
* CUDNN_OP_TENSOR_NOT:要执行的操作是求反，仅对A张量执行。

cudnnPoolingMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnPoolingMode_t是传递给
cudnnPoolingForward()、cudnnPoolingBackward()和cudnnSetPooling2dDescriptor()来选择要使用的池方法的枚举类型。 

* CUDNN_POOLING_MAX：使用池化窗口内的最大值。
* CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING：池化窗口内的值被平均。用于计算平均值的元素数量包括落在填充区域中的空间位置。
* CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING：池化窗口内的值被平均。用于计算平均值的元素数量不包括落在填充区域中的空间位置。
* CUDNN_POOLING_MAX_DETERMINISTIC：使用池化窗口内的最大值。使用的算法是确定性的。

cudnnReduceTensorIndices_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnReduceTensorIndices_t是一个枚举类型，用于指示cudnnReduceTensor()例程是否计算索引。
该枚举类型用作描述符的字段cudnnReduceTensorDescriptor_t。

* CUDNN_REDUCE_TENSOR_NO_INDICES：不计算指数。
* CUDNN_REDUCE_TENSOR_FLATTENED_INDICES：计算指数。得出的指数是相对的，并且是平坦的。

cudnnReduceTensorOp_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnReduceTensorOp_t是一个枚举类型，用于指示cudnnReduceTensor()例程要使用的 Tensor Core 操作。该枚举类型用作描述符的字段cudnnReduceTensorDescriptor_t。

* CUDNN_REDUCE_TENSOR_ADD：要执行的操作是加法。
* CUDNN_REDUCE_TENSOR_MUL：要执行的运算是乘法。
* CUDNN_REDUCE_TENSOR_MIN：要执行的操作是最小比较。
* CUDNN_REDUCE_TENSOR_MAX：要执行的操作是最大比较。
* CUDNN_REDUCE_TENSOR_AMAX：要执行的操作是绝对值的最大比较。
* CUDNN_REDUCE_TENSOR_AVG：要执行的操作是求平均。
* CUDNN_REDUCE_TENSOR_NORM1：要执行的运算是绝对值的加法。
* CUDNN_REDUCE_TENSOR_NORM2：要执行的运算是平方和的平方根。
* CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS：要执行的运算是乘法，不包括值为零的元素。

cudnnRNNAlgo_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnRNNAlgo_t是一个枚举类型，用于指定
cudnnRNNForwardInference()、cudnnRNNForwardTraining()、cudnnRNNBackwardData()和cudnnRNNBackwardWeights()例程中使用的算法。

* CUDNN_RNN_ALGO_STANDARD：每个 RNN 层都作为一系列操作来执行。该算法预计在广泛的网络参数范围内具有稳健的性能。
* CUDNN_RNN_ALGO_PERSIST_STATIC:网络的循环部分是使用持久内核方法执行的。当输入张量的第一维较小（即小批量）时，该方法预计会很快。
  CUDNN_RNN_ALGO_PERSIST_STATIC仅在计算能力 >= 6.0 的设备上受支持。
* CUDNN_RNN_ALGO_PERSIST_DYNAMIC:网络的循环部分是使用持久内核方法执行的。
  当输入张量的第一维较小（即小批量）时，该方法预计会很快。
  当使用CUDNN_RNN_ALGO_PERSIST_DYNAMIC持久内核时，在运行时准备并能够使用网络和活动 GPU 的特定参数进行优化。
  因此，在使用CUDNN_RNN_ALGO_PERSIST_DYNAMIC一次性计划时，必须执行准备阶段。然后可以在具有相同模型参数的重复调用中重用这些计划。

使用时支持的隐藏单元最大数量的限制CUDNN_RNN_ALGO_PERSIST_DYNAMIC明显高于使用时的限制CUDNN_RNN_ALGO_PERSIST_STATIC，
但是当超过 所支持的最大值时，吞吐量可能会显着降低CUDNN_RNN_ALGO_PERSIST_STATIC。
在这种情况下，这种方法在某些情况下仍然会表现出色CUDNN_RNN_ALGO_STANDARD。CUDNN_RNN_ALGO_PERSIST_DYNAMIC仅在Linux计算机上计算能力 >= 6.0 的设备上受支持。

cudnnSamplerType_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnSamplerType_t是传递给cudnnSpatialTfSamplerForward()、cudnnSpatialTfSamplerBackward()
和cudnnSetSpatialTransformerNdDescriptor()来选择要使用的采样器类型的枚举类型。 

* CUDNN_SAMPLER_BILINEAR：选择双线性采样器。

cudnnSeverity_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnSeverity_t是传递给用户可以设置的用于记录日志的自定义回调函数的枚举类型。
此枚举描述了项目的严重性级别，因此自定义日志记录回调可能会做出不同的反应。

* CUDNN_SEV_FATAL：该值表示 cuDNN 发出的致命错误。
* CUDNN_SEV_ERROR：该值表示 cuDNN 发出的正常错误。
* CUDNN_SEV_WARNING：该值表示 cuDNN 发出的警告。
* CUDNN_SEV_INFO：该值表示 cuDNN 发出的一条信息（例如 API 日志）。

cudnnSoftmaxAlgorithm_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnSoftmaxAlgorithm_t用于选择cudnnSoftmaxForward()和cudnnSoftmaxBackward()中使用的 softmax 函数的实现。

* CUDNN_SOFTMAX_FAST:该实现应用了简单的 softmax 运算。
* CUDNN_SOFTMAX_ACCURATE:此实现将 softmax 输入域的每个点缩放为其最大值，以避免 softmax 评估中潜在的浮点溢出。
* CUDNN_SOFTMAX_LOG:此条目执行 log softmax 操作，通过缩放输入域中的每个点来避免溢出，如CUDNN_SOFTMAX_ACCURATE中所示。

cudnnSoftmaxMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnSoftmaxMode_t用于选择cudnnSoftmaxForward()和cudnnSoftmaxBackward()正在计算其结果的数据。

* CUDNN_SOFTMAX_MODE_INSTANCE:softmax 运算是针对每个图像(N)的各个维度上(C,H,W)进行计算的。
* CUDNN_SOFTMAX_MODE_CHANNEL:softmax 运算是跨维度(H,W)按每个图像(N)的空间位置(C)计算的。


cudnnStatus_t
~~~~~~~~~~~~~~~~~~~~~~~~
cudnnStatus_t是用于函数状态返回的枚举类型。所有 cuDNN 库函数都会返回其状态，可以是以下值之一

* CUDNN_STATUS_SUCCESS:成功
* CUDNN_STATUS_NOT_INITIALIZED：cuDNN 库未正确初始化。
  当调用cudnnCreate()失败或在cudnnCreate()调用另一个 cuDNN 例程之前尚未调用时，通常会返回此错误。
  cudnnCreate()在前一种情况下，通常是由于硬件设置中的错误 调用的 CUDA 运行时 API 中出现错误。
* CUDNN_STATUS_ALLOC_FAILED：cuDNN 库内部资源分配失败。这通常是由内部cudaMalloc()故障引起的。
  为了更正，在函数调用之前，尽可能地释放先前分配的内存。
* CUDNN_STATUS_BAD_PARAM：向函数传递了不正确的值或参数。要更正，请确保传递的所有参数都具有有效值。
* CUDNN_STATUS_ARCH_MISMATCH：该函数需要当前 GPU 设备所没有的功能。请注意，cuDNN 仅支持计算能力大于或等于 3.0 的设备。
  要更正，请在具有适当计算能力的设备上编译并运行应用程序。
* CUDNN_STATUS_MAPPING_ERROR：访问GPU内存空间失败，通常是由于纹理绑定失败造成的。
  要更正，请在函数调用之前取消绑定任何先前绑定的纹理。否则，这可能表明库中存在内部错误/错误。
* CUDNN_STATUS_EXECUTION_FAILED：GPU程序执行失败。这通常是由于在GPU上启动某些cuDNN内核失败造成的，可能有多种原因。
  要进行更正，请检查硬件、适当版本的驱动程序以及 cuDNN 库是否已正确安装。否则，这可能表明库中存在内部错误/错误。
* CUDNN_STATUS_INTERNAL_ERROR：内部 cuDNN 操作失败。
* CUDNN_STATUS_NOT_SUPPORTED：cuDNN 目前不支持所请求的功能。
* CUDNN_STATUS_LICENSE_ERROR：请求的功能需要一些许可证，并且在尝试检查当前许可时检测到错误。如果许可证不存在或已过期，或者环境变量NVIDIA_LICENSE_FILE设置不正确，则可能会发生此错误。
* CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING：在预定义的搜索路径中找不到 cuDNN 所需的运行时库。这些库是libcuda.so( nvcuda.dll) 和libnvrtc.so(nvrtc64_<Major Release Version><Minor Release Version>_0.dll和nvrtc-builtins64_<Major Release Version><Minor Release Version>.dll)。
* CUDNN_STATUS_RUNTIME_IN_PROGRESS：用户流中的某些任务未完成。
* CUDNN_STATUS_RUNTIME_FP_OVERFLOW：GPU内核执行期间发生数值溢出。


cudnnTensorFormat_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnTensorFormat_t是一个枚举类型，被cudnnSetTensor4dDescriptor()用于创建一个预定义布局的张量。
关于这些张量如何在内存中排列的详细解释，请参阅数据布局格式。

* CUDNN_TENSOR_NCHW：此张量格式指定数据按以下顺序布局：批量大小、特征图、行、列。
  步幅是隐式定义的，数据在内存中是连续的，图像、特征图、行和列之间没有填充；列是内部维度，图像是最外部维度。
* CUDNN_TENSOR_NHWC：这种张量格式规定数据按以下顺序排列：批量大小、行、列、特征图。数据在内存中是连续的，图像、行、列和特征图之间没有填充；特征图是内层维度，图像是最外层维度。
* CUDNN_TENSOR_NCHW_VECT_C：此张量格式指定数据按以下顺序布局：批量大小、特征图、行、列。
  然而，张量的每个元素都是多个特征图的向量。向量的长度由张量的数据类型携带。
  步幅是隐式定义的，数据在内存中是连续的，图像、特征图、行和列之间没有填充；列是内部维度，图像是最外部维度。
  此格式仅支持张量数据类型CUDNN_DATA_INT8x4、CUDNN_DATA_INT8x32和CUDNN_DATA_UINT8x4。
  也CUDNN_TENSOR_NCHW_VECT_C可以按以下方式解释：NCHW INT8x32 格式实际上是 N x (C/32) x H x W x 32（每个 W 32 个 C），
  就像 NCHW INT8x4 格式是 N x (C/4) ) x H x W x 4（每 W 4 个 C）。
  因此，VECT_C名称 - 每个 W 都是 Cs 的向量（4 或 32）。

API函数
``````````

cudnnActivationForward()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
对每个输入值逐元素应用指定的神经元激活函数。

该例程允许就地操作；也就是说，xData和yData指针可以相等。
但是，这要求xDesc和yDesc描述符完全相同（特别是，输入和输出的步长必须匹配，才允许就地操作）

支持所有4 维和 5 维张量格式，但是，当xDesc步幅等于yDesc步幅时，可以获得最佳性能HW-packed。对于超过 5 个维度，张量必须压缩其空间维度。

.. code-block:: cpp

    cudnnStatus_t cudnnActivationForward(
        cudnnHandle_t handle,
        cudnnActivationDescriptor_t     activationDesc,
        const void                     *alpha,
        const cudnnTensorDescriptor_t   xDesc,
        const void                     *x,
        const void                     *beta,
        const cudnnTensorDescriptor_t   yDesc,
        void                           *y);
    // handle:cuDNN上下文句柄
    // activationDesc 激活函数描述符
    // alpha, beta:指向缩放因子（在主机内存中）的指针，用于将计算结果与输出层中的先验值混合
    //         dstValue = alpha[0]*result + beta[0]*priorDstValue
    // xDesc:先前初始化的输入张量描述符的句柄
    // x:指向与xDesc张量描述符关联的 GPU 内存的数据指针
    // yDesc:先前初始化的输出张量描述符的句柄。
    // y:指向与yDesc张量描述符关联的 GPU 内存的数据指针

cudnnAddTensor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
此函数将bias张量的缩放值添加到另一个张量。bias张量的每个维度A必须与目标张量的相应维度C匹配，或者必须等于 1。
在后一种情况下，这些维度的bias张量中的相同值将用于混合到张量C中。

仅支持 4D 和 5D 张量。超出这些尺寸，则不支持此例程。

.. code-block:: cpp

    cudnnStatus_t cudnnAddTensor(
        cudnnHandle_t                     handle,
        const void                       *alpha,
        const cudnnTensorDescriptor_t     aDesc,
        const void                       *A,
        const void                       *beta,
        const cudnnTensorDescriptor_t     cDesc,
        void                             *C);
    // alpha, beta:指向缩放因子（在主机内存中）的指针，用于将计算结果与输出层中的先验值混合
    //         dstValue = alpha[0]*result + beta[0]*priorDstValue


cudnnBatchNormalizationForwardInference()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数执行推理阶段的前向批量归一化层计算。该层基于批量归一化：通过减少内部协变量偏移来加速深度网络训练论文。

仅支持4D和5D张量。该函数执行的输入变换定义如下:

`y = beta*y + alpha *[bnBias + (bnScale * (x-estimatedMean)/sqrt(epsilon + estimatedVariance)]`

.. code-block:: cpp

    cudnnStatus_t cudnnBatchNormalizationForwardInference(
        cudnnHandle_t                    handle,
        cudnnBatchNormMode_t             mode,
        const void                      *alpha,
        const void                      *beta,
        const cudnnTensorDescriptor_t    xDesc,
        const void                      *x,
        const cudnnTensorDescriptor_t    yDesc,
        void                            *y,
        const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
        const void                      *bnScale,
        const void                      *bnBias,
        const void                      *estimatedMean,
        const void                      *estimatedVariance,
        double                           epsilon)
    // mode:操作模式
    // alpha, beta:指向缩放因子（在主机内存中）的指针，用于将计算结果与输出层中的先验值混合
    //         dstValue = alpha[0]*result + beta[0]*priorDstValue
    // bnScaleBiasMeanVarDesc, bnScale,bnBias:
    //  设备内存中用于批量归一化标度和偏置参数的张量描述符和指针（在批量归一化论文中：偏置被称为β，标度被称为γ）
    // estimatedMean,estimatedVariance:均值和方差张量（与偏置和标度具有相同的描述符）。
    //   调用 cudnnBatchNormalizationForwardTraining()时在训练阶段积累的resultRunningMean（平均值）和 
    //   resultRunningVariance（方差）应作为输入在此处传递。
    // epsilon:用于批量标准化公式中的ε值。它的值应该等于或大于在cudnn.h中为CUDNN_BN_MIN_EPSILON定义的值。

该函数支持以下各种描述符的数据类型组合:

+--------------------------+---------------------+------------------------+-------------------+---------------------+
| Data Type Configurations |        xDesc        | bnScaleBiasMeanVarDesc |    alpha, beta    |        yDesc        |
+==========================+=====================+========================+===================+=====================+
| INT8_CONFIG              | CUDNN_DATA_INT8     | CUDNN_DATA_FLOAT       | CUDNN_DATA_FLOAT  | CUDNN_DATA_INT8     |
+--------------------------+---------------------+------------------------+-------------------+---------------------+
| PSEUDO_HALF_CONFIG       | CUDNN_DATA_HALF     | CUDNN_DATA_FLOAT       | CUDNN_DATA_FLOAT  | CUDNN_DATA_HALF     |
+--------------------------+---------------------+------------------------+-------------------+---------------------+
| FLOAT_CONFIG             | CUDNN_DATA_FLOAT    | CUDNN_DATA_FLOAT       | CUDNN_DATA_FLOAT  | CUDNN_DATA_FLOAT    |
+--------------------------+---------------------+------------------------+-------------------+---------------------+
| DOUBLE_CONFIG            | CUDNN_DATA_DOUBLE   | CUDNN_DATA_DOUBLE      | CUDNN_DATA_DOUBLE | CUDNN_DATA_DOUBLE   |
+--------------------------+---------------------+------------------------+-------------------+---------------------+
| BFLOAT16_CONFIG          | CUDNN_DATA_BFLOAT16 | CUDNN_DATA_FLOAT       | CUDNN_DATA_FLOAT  | CUDNN_DATA_BFLOAT16 |
+--------------------------+---------------------+------------------------+-------------------+---------------------+

cudnnCreate()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数初始化cuDNN库，并创建一个不透明结构的句柄来保存cuDNN库上下文。
它分配主机和设备上的硬件资源，必须在调用其他cuDNN库之前调用。

cuDNN库句柄与当前CUDA设备（上下文）绑定。要在多个设备上使用该库，需要为每个设备创建一个cuDNN句柄。

对于一个给定的设备，可能会创建多个具有不同配置（例如，不同的当前CUDA流）的cuDNN句柄。
由于cudnnCreate()分配了一些内部资源，通过调用cudnnDestroy()释放这些资源将隐式调用cudaDeviceSynchronize()；
因此，推荐的最佳做法是在性能关键代码路径之外调用cudnnCreate/cudnnDestroy。

对于从不同线程使用同一设备的多线程应用程序，建议的编程模型是为每个线程创建一个（或几个，如果方便的话）cuDNN 句柄，
并在线程的整个生命周期中使用该 cuDNN 句柄。

.. code-block:: cpp

    cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);

cudnnCreateActivationDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配不透明结构所需的内存来创建激活描述符对象。更多信息，请参阅cudnnActivationDescriptor_t。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateActivationDescriptor(
        cudnnActivationDescriptor_t   *activationDesc);

cudnnCreateAlgorithmPerformance()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配保存其不透明结构所需的内存来创建多个算法性能对象。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateAlgorithmPerformance(
        cudnnAlgorithmPerformance_t *algoPerf,
        int                         numberToCreate);

cudnnCreateDropoutDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配不透明结构所需的内存来创建一个通用的Dropout描述符对象。更多信息，请参阅cudnnDropoutDescriptor_t。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateDropoutDescriptor(
        cudnnDropoutDescriptor_t    *dropoutDesc);

cudnnCreateFilterDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配不透明结构所需的内存来创建过滤器描述符对象。更多信息，请参阅cudnnFilterDescriptor_t。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateFilterDescriptor(
        cudnnFilterDescriptor_t *filterDesc);

cudnnCreateLRNDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数分配用于保存LRN和DivisiveNormalization层操作所需数据的内存，并返回用于后续层向前和向后调用的描述符。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateLRNDescriptor(
            cudnnLRNDescriptor_t    *poolingDesc);

cudnnCreateOpTensorDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数用于创建张量数学描述符。更多信息，请参阅cudnnOpTensorDescriptor_t。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateOpTensorDescriptor(
            cudnnOpTensorDescriptor_t* 	opTensorDesc);

cudnnCreatePoolingDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配不透明结构所需的内存来创建一个池描述符对象。

.. code-block:: cpp

    cudnnStatus_t cudnnCreatePoolingDescriptor(
            cudnnPoolingDescriptor_t    *poolingDesc);

cudnnCreateReduceTensorDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: cpp

    cudnnStatus_t cudnnCreateReduceTensorDescriptor(
	    cudnnReduceTensorDescriptor_t*	reduceTensorDesc);

cudnnCreateSpatialTransformerDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: cpp

    cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(
        cudnnSpatialTransformerDescriptor_t *stDesc);

cudnnCreateTensorDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配保存其不透明结构所需的内存来创建通用张量描述符对象。数据被初始化为全零

.. code-block:: cpp

    cudnnStatus_t cudnnCreateTensorDescriptor(
                cudnnTensorDescriptor_t *tensorDesc);

cudnnCreateTensorTransformDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配保存其不透明结构所需的内存来创建张量变换描述符对象。张量数据被初始化为全零。
使用cudnnSetTensorTransformDescriptor()该函数来初始化由该函数创建的描述符。

.. code-block:: cpp

    cudnnStatus_t cudnnCreateTensorTransformDescriptor(
	        cudnnTensorTransformDescriptor_t *transformDesc);

cudnnDeriveBNTensorDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数从图层的 x 数据描述符中导出批归一化scale、invVariance、bnBias 和 bnScale 子张量的辅助张量描述符。

.. code-block:: cpp

    cudnnStatus_t cudnnDeriveBNTensorDescriptor(
        cudnnTensorDescriptor_t         derivedBnDesc,
        const cudnnTensorDescriptor_t   xDesc,
        cudnnBatchNormMode_t            mode);

cudnnDeriveNormTensorDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数根据图层的 x 数据描述符和常模为归一化均值、不变性、normBias 和 normScale 子张量导出张量描述符。
归一化、均值和不变性共享相同的描述符，而偏置和标度共享相同的描述符。

.. code-block:: cpp
  
    cudnnStatus_t CUDNNWINAPI
    cudnnDeriveNormTensorDescriptor(cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
                                    cudnnTensorDescriptor_t derivedNormMeanVarDesc,
                                    const cudnnTensorDescriptor_t xDesc,
                                    cudnnNormMode_t mode,
                                    int groupCnt);

cudnnDestroy()
~~~~~~~~~~~~~~~~~~~~~~~~~~
释放cuDNN句柄使用的资源。该函数通常是使用特定句柄对cuDNN进行的最后一次调用。
由于cudnnCreate()分配内部资源，调用cudnnDestroy()释放这些资源将隐式调用cudaDeviceSynchronize()；
因此，推荐的最佳做法是在性能关键代码路径之外调用cudnnCreate/cudnnDestroy。

.. code-block:: cpp

    cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnSetFilter4dDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
此函数将先前创建的滤波器描述符对象初始化为 4D 滤波器。滤波器的布局必须在内存中连续。

.. code-block:: cpp

  cudnnStatus_t cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t    filterDesc,
    cudnnDataType_t            dataType,
    cudnnTensorFormat_t        format,
    int                        k,
    int                        c,
    int                        h,
    int                        w)
  // k: 输出特征图的数量
  // c:输入特征图的数量
  // h,w:滤波器的高度和宽度

cudnnSetFilterNdDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数初始化之前创建的滤波器描述符对象。滤波器的布局必须在内存中连续。

.. code-block:: cpp

  cudnnStatus_t cudnnSetFilterNdDescriptor(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t         dataType,
    cudnnTensorFormat_t     format,
    int                     nbDims,
    const int               filterDimA[])
  // nbDims:滤波器的维度
  // filterDimA:滤波器各维度的大小


cudnn_cnn_infer.so库
--------------------------------------

指向Opaque结构体的指针
```````````````````````````````````
cudnnConvolutionDescriptor_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnConvolutionDescriptor_t 是一个指向不透明结构的指针，该结构保存卷积操作的描述。
cudnnCreateConvolutionDescriptor() 用于创建一个实例，而 cudnnSetConvolutionNdDescriptor() 或 
cudnnSetConvolution2dDescriptor() 用于初始化该实例。


cudnnConvolutionFwdAlgoPerf_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnConvolutionFwdAlgoPerf_t 是一个结构，包含 cudnnFindConvolutionForwardAlgorithm() 返回的性能结果或 
cudnnGetConvolutionForwardAlgorithm_v7() 返回的启发式性能结果。

成员变量：

* cudnnConvolutionFwdAlgo_t algo：该算法运行后可获得相关的性能指标
* cudnnStatus_t status：
* float time：cudnnConvolutionForward() 的执行时间（毫秒）
* size_t memory：工作区大小（以字节为单位）。
* cudnnDeterminism_t determinism：算法的确定性
* cudnnMathType_t mathType：为算法提供的数学类型。
* int reserved[3]：为未来特性预留空间。


枚举类型
`````````````````````
cudnnConvolutionFwdAlgo_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnConvolutionFwdAlgo_t 是一种枚举类型，它公开了可用于执行正向卷积操作的不同算法。

* CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:这种算法将卷积表述为矩阵乘积，而实际上并没有明确形成容纳输入张量数据的矩阵。
* CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:这种算法将卷积表达为矩阵乘积，而不需要明确形成容纳输入张量数据的矩阵，但仍需要一些内存工作空间来预先计算一些索引，以便于隐式构建容纳输入张量数据的矩阵。
* CUDNN_CONVOLUTION_FWD_ALGO_GEMM:该算法将卷积表示为显式矩阵乘积。需要大量的内存工作空间来存储保存输入张量数据的矩阵。
* CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:该算法将卷积表示为直接卷积（例如，不隐式或显式地进行矩阵乘法）。
* CUDNN_CONVOLUTION_FWD_ALGO_FFT:该算法使用快速傅立叶变换方法来计算卷积。需要大量的内存工作空间来存储中间结果。
* CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:该算法采用快速傅里叶变换方法，但将输入分割成块。存储中间结果需要相当大的内存工作空间，但对于大尺寸图像来说，比 CUDNN_CONVOLUTION_FWD_ALGO_FFT 要小。
* CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:该算法使用 Winograd 变换方法来计算卷积。需要一个合理大小的工作空间来存储中间结果。
* CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:该算法使用 Winograd 变换法计算卷积。可能需要很大的工作空间来存储中间结果。


cudnnConvolutionMode_t
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudnnConvolutionMode_t是cudnnSetConvolution2dDescriptor()用于配置卷积描述符的枚举类型。
用于卷积的滤波器可以以两种不同的方式应用，在数学上对应于卷积或互相关。（互相关相当于滤波器旋转 180 度的卷积。）

* CUDNN_CONVOLUTION:在此模式下，将滤波器应用于图像时将完成卷积运算。
* CUDNN_CROSS_CORRELATION:在此模式下，将滤波器应用于图像时将进行互相关操作。


API函数
```````````````````````````

cudnnCreateConvolutionDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数通过分配opaque结构所需的内存来创建卷积描述符对象。

.. code-block:: cpp

  cudnnStatus_t cudnnCreateConvolutionDescriptor(
    cudnnConvolutionDescriptor_t *convDesc)

cudnnSetConvolution2dDescriptor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
此函数将先前创建的卷积描述符对象初始化为二维相关对象。
此函数假定张量和滤波器描述符对应于前向卷积路径，并检查其设置是否有效。
相同的卷积描述符可以在后向路径中重复使用，前提是它对应于相同的层。

.. code-block:: cpp
 
  cudnnStatus_t cudnnSetConvolution2dDescriptor(
      cudnnConvolutionDescriptor_t    convDesc,
      int                             pad_h,
      int                             pad_w,
      int                             u,
      int                             v,
      int                             dilation_h,
      int                             dilation_w,
      cudnnConvolutionMode_t          mode,
      cudnnDataType_t                 computeType)
  // pad_h:Zero-padding height：输入图像顶部和底部的零的行数。
  // pad_w:Zero-padding width:输入图像左侧和右侧隐式连接的零列数。
  // u:kernel的垂直步长
  // v:kernel的水平步长
  // dilation_h:滤波器高度扩张
  // dilation_w:滤波器宽度扩张
  // mode:CUDNN_CONVOLUTION 或 CUDNN_CROSS_CORRELATION
  // computeType:计算精度


cudnnGetConvolution2dForwardOutputDim()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
给定卷积描述符、输入张量描述符和滤波器描述符，此函数返回 2D 卷积所得 4D 张量的维度。
此函数可以帮助设置输出张量并在启动实际卷积之前分配适当的内存

.. code-block:: cpp

  cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t  convDesc,
    const cudnnTensorDescriptor_t       inputTensorDesc,
    const cudnnFilterDescriptor_t       filterDesc,
    int                                *n,
    int                                *c,
    int                                *h,
    int                                *w)
    // convDesc:卷积描述符
    // inputTensorDesc:输入向量描述符
    // filterDesc:滤波器描述符
    // n,c,h,w:输出维度

h和w使用以下公式计算：outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride

cudnnGetConvolutionForwardAlgorithmMaxCount()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数返回算法的最大个数，包含一般算法和TensorCore算法

.. code-block:: cpp

  cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(
    cudnnHandle_t   handle,
    int             *count);

cudnnGetConvolutionForwardAlgorithm_v7()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
该函数是一种启发式方法，用于为 cudnnConvolutionForward() 获取最适合给定层规范的算法。
按预期（基于内部启发式）相对性能排序，最快的算法为 perfResults 的索引 0。
如需详尽搜索最快算法，请使用 cudnnFindConvolutionForwardAlgorithm()。可以通过returnedAlgoCount变量查询结果算法的总数。

.. code-block:: cpp

  cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t                       handle,
    const cudnnTensorDescriptor_t       xDesc,
    const cudnnFilterDescriptor_t       wDesc,
    const cudnnConvolutionDescriptor_t  convDesc,
    const cudnnTensorDescriptor_t       yDesc,
    const int                           requestedAlgoCount,
    int                                *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t      *perfResults)
  // requestedAlgoCount:存储在 perfResults 中的元素的最大数量
  // returnedAlgoCount:存储在 perfResults 中的输出元素数量
  // perfResults:用户分配的数组，用于存储按计算时间升序排序的性能指标。
  

cudnnGetConvolutionForwardWorkspaceSize()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
此函数返回用户在使用指定算法调用 cudnnConvolutionForward() 时需要分配的 GPU 内存工作空间。
分配的工作区将传递给例程 cudnnConvolutionForward()。
指定的算法可以是调用 cudnnGetConvolutionForwardAlgorithm_v7() 的结果，也可以由用户任意选择。
请注意，并非每种算法都适用于输入张量的每种配置和/或卷积描述符的每种配置。

.. code-block:: cpp

  cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t   handle,
    const   cudnnTensorDescriptor_t         xDesc,
    const   cudnnFilterDescriptor_t         wDesc,
    const   cudnnConvolutionDescriptor_t    convDesc,
    const   cudnnTensorDescriptor_t         yDesc,
    cudnnConvolutionFwdAlgo_t               algo,
    size_t                                 *sizeInBytes)
  // algo:指定所选卷积算法的枚举。
  // sizeInBytes:使用指定算法执行正向卷积所需的 GPU 工作空间内存量

cudnnConvolutionForward()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
此函数使用 w 指定的滤波器对 x 执行卷积或交叉相关运算，返回结果为 y。

.. code-block:: cpp

  cudnnStatus_t cudnnConvolutionForward(
      cudnnHandle_t                       handle,
      const void                         *alpha,
      const cudnnTensorDescriptor_t       xDesc,
      const void                         *x,
      const cudnnFilterDescriptor_t       wDesc,
      const void                         *w,
      const cudnnConvolutionDescriptor_t  convDesc,
      cudnnConvolutionFwdAlgo_t           algo,
      void                               *workSpace,
      size_t                              workSpaceSizeInBytes,
      const void                         *beta,
      const cudnnTensorDescriptor_t       yDesc,
      void                               *y)

cudnnAddTensor()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
此函数将偏置张量的缩放值添加到另一个张量中。
偏置张量 A 的每个维度必须与目标张量 C 的相应维度相匹配，或者必须等于 1。
在后一种情况下，偏置张量中这些维度的相同值将被用于混合到 C 张量中。

仅支持 4D 和 5D 张量。超出这两个维度，则不支持此例程。

.. code-block:: cpp

  cudnnStatus_t cudnnAddTensor(
      cudnnHandle_t                     handle,
      const void                       *alpha,
      const cudnnTensorDescriptor_t     aDesc,
      const void                       *A,
      const void                       *beta,
      const cudnnTensorDescriptor_t     cDesc,
      void                             *C)












