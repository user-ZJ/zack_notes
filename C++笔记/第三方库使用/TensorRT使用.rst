TensorRT使用
=====================

cudnn安装
----------------

.. code-block:: shell

  sudo cp cuda/include/cudnn*.h /usr/local/cuda/include  
  sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64  
  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 




tensorrt安装
--------------------

https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

下载：
* https://developer.nvidia.com/nvidia-tensorrt-7x-download
* https://developer.nvidia.com/nvidia-tensorrt-8x-download

.. code-block:: shell

  version="7.x.x.x"
  os="<os>"
  arch=$(uname -m)
  cuda="cuda-x.x"
  cudnn="cudnn8.x"
  tar xzvf TensorRT-${version}.${os}.${arch}-gnu.${cuda}.${cudnn}.tar.gz
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
  cd <TensorRT-${version}/samples
  make




pytorch转tensorrt
-------------------------
.. image:: /images/pytorch-tensorrt.png
  :align: center

方式1：加载onnx模型文件，使用tensorrt的build在程序运行的时候根据当前平台构建tensorrt的Engine； **好处** 是所有nvidia的平台均适用； **坏处** 是每次构建模型的时候很耗时

方式2：使用trtexec工具将onnx预先构建为tensorrt的Engine，并通过二进制方式保存，加载的时候直接加载tensorrt的Engine； **好处** 是每次加载很快； **坏处** 是在什么平台上(如V100)上构建的模型，只能在该平台上使用。

trtexec工具
------------------

.. code-block:: shell

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dmai/TensorRT-7.2.3.4/lib:/usr/local/cuda/lib64
  # 动态维度，在转onnx的时候需要将对应维度通过dynamic_axes设置为动态维度
  trtexec --explicitBatch --onnx=pse_sim.onnx \
    --fp16  \
    --minShapes=input:1x3x100x100 \
    --optShapes=input:1x3x896x1312 \
    --maxShapes=input:1x3x2000x3000 \
    --shapes=input:1x3x1000x1000 \
    --saveEngine=pse_sim.engine



多线程
--------------

多线程示例：

:download:`TensorRT_sample.zip </downloads/TensorRT_sample.zip>`

::

  1. Please prepare TensorRT engine of GPU and DLA with trtexec first. For example, 
  $ /usr/src/tensorrt/bin/trtexec --onnx=/usr/src/tensorrt/data/mnist/mnist.onnx --saveEngine=gpu.engine 
  $ /usr/src/tensorrt/bin/trtexec --onnx=/usr/src/tensorrt/data/mnist/mnist.onnx --useDLACore=0 \
    --allowGPUFallback --saveEngine=dla.engine 
  2. Compile TensorRT_sample.zip (3.5 KB) 
  $ unzip TensorRT_sample.zip 
  $ cd TensorRT_sample 
  $ nvcc -std=c++11 main.cpp -I/root/TensorRT-7.2.2.3/include/ -L/root/TensorRT-7.2.2.3/lib \
     -lnvinfer -o test
  3. Test Please…

DLA
---------------
TensorRT DLA是指TensorRT库中的Deep Learning Accelerator（深度学习加速器）功能，它可以利用NVIDIA的Deep Learning Accelerator（DLA）硬件加速器来加速深度学习推理。DLA是一种专门为深度学习推理而设计的硬件加速器，它可以在低功耗和低延迟的情况下提供高性能的推理加速。
TensorRT DLA功能可以在支持DLA硬件的NVIDIA GPU上使用，它可以将深度学习模型的某些层（例如卷积层和全连接层）映射到DLA硬件上进行加速，从而提高推理性能。在使用TensorRT DLA功能时，可以通过TensorRT API或者TensorRT Python API来配置和控制DLA硬件的使用。

需要注意的是，TensorRT DLA功能目前仅支持一些特定的NVIDIA GPU型号，目前，支持DLA硬件加速器的GPU主要包括以下几种：

* NVIDIA Jetson AGX Xavier：Jetson AGX Xavier是一款面向AI和机器人应用的嵌入式平台，它搭载了NVIDIA自主研发的Volta GPU和两个DLA硬件加速器。
* NVIDIA Jetson Xavier NX：Jetson Xavier NX是一款面向AI和机器人应用的嵌入式平台，它搭载了NVIDIA自主研发的Volta GPU和一个DLA硬件加速器。

需要注意的是，支持DLA硬件加速器的GPU型号可能会随着硬件和软件的更新而发生变化。在使用TensorRT DLA功能时，需要查看所使用的GPU型号是否支持DLA硬件加速器，并且已经安装了相应的驱动程序和软件库。

cudaStream和cudaEvent
------------------------------
cudaStream和cudaEvent都是CUDA中的异步编程工具，用于实现GPU计算的并行化和优化。它们的主要区别在于它们的作用和使用方式不同。

cudaStream是一种用于管理GPU计算任务的工具，它可以将多个GPU计算任务组织成一个队列，然后按照队列中的顺序依次执行。通过使用
cudaStream，可以将多个GPU计算任务并行化执行，从而提高GPU计算的效率。
cudaStream的使用方式比较简单，可以通过cudaStreamCreate函数创建一个cudaStream
对象，然后将GPU计算任务提交到cudaStream中执行。

cudaEvent是一种用于测量GPU计算时间的工具，它可以在GPU计算任务开始和结束时记录时间戳，并计算出GPU计算任务的执行时间。通过使用
cudaEvent，可以对GPU计算任务的性能进行评估和优化。
cudaEvent的使用方式比较复杂，需要使用cudaEventCreate函数创建一个cudaEvent对象，然后使用
cudaEventRecord函数记录时间戳，最后使用cudaEventSynchronize函数等待GPU计算任务执行完成，并使用
cudaEventElapsedTime函数计算GPU计算任务的执行时间。

需要注意的是，cudaStream和cudaEvent都是异步编程工具，它们可以在GPU计算任务执行的同时进行其他操作，例如CPU计算、数据传输等。在使用
cudaStream和cudaEvent时，需要注意管理好GPU计算任务的执行顺序和时间，以避免出现不必要的延迟和竞争条件。


启用Plugin
-----------------------
添加头文件 "NvInferPlugin.h"

在代码中添加：

  initLibNvInferPlugins(&gLogger, "");

链接时添加nvinfer_plugin库

FAQ
-------------

1. **[F] [TRT] Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS**

   解决方法：添加--tacticSources=-cublasLt,+cublas选项

   参考：https://github.com/NVIDIA/TensorRT/issues/866



参考
---------

https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/

https://forums.developer.nvidia.com/t/how-to-use-tensorrt-by-the-multi-threading-package-of-python/123085

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html

https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#introductory_parser_samples