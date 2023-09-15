windows系统调试
=========================

windows下创建cuda工程(Visual Studio 2017)
-----------------------------------------------------
1. 打开Visual Studio,文件->新建->项目

.. image:: /images/cuda编程/cuda_windows1.jpg

2. 创建项目后，在导航栏 工具->选项->文本编辑器->文件扩展名，新增扩展名.cu，并将编辑器设置为Microsoft Visual C++

.. image:: /images/cuda编程/cuda_windows2.jpg

3. 工具->选项->项目和解决方案->VC++项目设置，添加要包括的扩展名.cu

.. image:: /images/cuda编程/cuda_windows3.jpg

4. 右键打开的项目->生成依赖项->生成自定义->勾选CUDA

.. image:: /images/cuda编程/cuda_windows4.jpg

5. 右键.cu文件->文件属性，设置为CUDA c/c++

.. image:: /images/cuda编程/cuda_windows5.jpg

1. 右键打开的项目->属性,配置计算能力

.. image:: /images/cuda编程/cuda_windows6.jpg