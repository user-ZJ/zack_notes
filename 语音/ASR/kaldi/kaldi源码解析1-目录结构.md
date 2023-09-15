# kaldi源码解析1-目录结构

kaldi源码存放在github上，github地址为https://github.com/kaldi-asr/kaldi

本文章源码解析截止到commit id 为ec83d38fcce88cc7c34ff81c857e4e8cb004182c的提交

源码目录目录说明

| 目录                 | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| base                 | kaldi基础类，以及version控制相关头文件                       |
| bin                  | kaldi可执行文件源码                                          |
| chain/chainbin       | chain模型基础类                                              |
| configure            | 编译控制脚本                                                 |
| cudadecoder          | GPU解码器，CPU decoder下功能的部分实现                       |
| cudafeat/cudafeatbin | GPU提特征的，CPU提特征的部分实现                             |
| cudamatrix           | GPU计算库，包括matrix和vector计算                            |
| decoder              | CPU解码器                                                    |
| doc                  | 文档                                                         |
| feat/featbin         | CPU特征提取                                                  |
| fgmmbin              | full-covariance GMM 模型bin                                  |
| fstbin               | FST 扩展                                                     |
| fstext               | FST 扩展                                                     |
| gst-plugin           | GStreamer插件，用于online解码                                |
| hmm                  | HMM模型                                                      |
| itf                  | 扩展接口，比如 OptimizableInterface， OnlineFeatureInterface 等 |
| ivector/ivectorbin   | ivector基础代码                                              |
| kws/kwsbin           | Keyword Search                                               |
| lat/latbin           | lattice（词格）相关                                          |
| lm/lmbin             | Language Model（语言模型）                                   |
| matrix               | CPU矩阵运算库，支持mkl，openblas等加速                       |
| nnet/nnetbin         | 第一代网络实现，支持单GPU训练                                |
| nnet2/nnet2bin       | nnet1的重构版本，支持多GPU并行训练                           |
| nnet3/nnet3bin       | 第三代网络，采用基于计算图构建的网络定义方式，目前主流       |
| online/onlinebin     | 在线解析                                                     |
| online2/online2bin   | 在线解析                                                     |
| probe                | exp性能测试工具                                              |
| rnnlm/rnnlmbin       | 基于 rnn 语言模型                                            |
| sgmm2/sgmm2bin       | SGMM (子空间高斯混合) 模型                                   |
| tfrnnlm/tfrnnlmbin   | 基于 tensorflow rnn 语言模型                                 |
| transform            | 特征转换                                                     |
| tree                 | 内部决策树                                                   |
| util                 | 基础工具                                                     |



## **kaldi/matrix**

1. kaldi-blas: 定义使用哪个blas
2. matrix-comm: 定义了基本的矩阵类型
3. packed-matrix : 基本压缩矩阵，声明模板类PackedMatrix
4. tp-matrix 和 sp-matrix: 三角矩阵和对称矩阵TpMatrix SpMatrix，其基类是PackedMatrix
5. kaldi-vector 和 kaldi-matrix: kaldi中的向量和矩阵，声明模板类 VectorBase/Vector SubVector/MatrixBase/SubMatrix Matrix
6. jama-eig jama-svd: 特征分解和奇异值分解，只有使用atlas时才用到这两个文件，因为其他的库已经自带这两个算法了
7. matrix-functions: 矩阵计算应用函数，如计算FFT