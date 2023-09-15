GMM高斯混合模型详解
====================

高斯模型
-----------
一维高斯模型公式为：

.. math::
  \eta(x|\mu ,\sigma )=\frac{1}{\sigma \sqrt{2\pi } } \exp (-\frac{(x-\mu)^{2} }{2\sigma ^{2}} )

其中 :math:`\mu` 为均值， :math:`\sigma` 为方差。

多维高斯模型公式为：

.. math:: 
  \eta(x|\mu ,\Sigma )=\frac{1}{(\sqrt{(2\pi)^{D}|\Sigma| }) } \exp (-\frac{(x-\mu)^{T} \Sigma^{-1} (x-\mu)}{2})

其中 :math:`\mu` 为均值向量， :math:`\Sigma` 为协方差矩阵，D为数据维度。

高斯混合模型
--------------
假设混合高斯模型由K个高斯模型组成。
一维高斯混合模型可以表示为:

.. math:: 
  P(x) = \sum_{k=1}^{K}\alpha_{k} \eta (x|\mu_{k},\sigma_{k});  \sum_{k=1}^{K}\alpha_{k}=1, \alpha_{k}>0

其中 :math:`\alpha_{k}` 是观测数据属于第k个子模型的概率

多维混合高斯模型可以表示为： 

.. math:: 
    P(x) = \sum_{k=1}^{K}\alpha_{k} \eta (x|\mu_{k},\Sigma_{k});  \sum_{k=1}^{K}\alpha_{k}=1, \alpha_{k}>0

其中 :math:`\alpha_{k}` 是观测数据属于第k个子模型的概率

使用EM算法优化GMM
----------------------
一维混合高斯模型优化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
初始化
```````````
1. 使用样本数据X= :math:`{x_1,...,x_N}` 对高斯模型均值 :math:`\mu_1,...\mu_K` 进行随机初始化。如：3个高斯模型，100个样本点，设置 :math:`\mu_1=x_{45},\mu_2=x_{32},\mu_3=x_{10}` 
2. 将高斯模型方差设置为样本方差。 :math:`\sigma_1,...,\sigma_K=\frac{1}{N}\sum_{i=1}^{N}(x_i-\bar{x})^2` , :math:`\bar{x}` 是样本的均值， :math:`\bar{x}=\frac{1}{N}\sum_{i=1}^{N}x_i` 
3. 将所有 :math:`\alpha_{k}` 设置为一个均匀分布，即 :math:`\alpha_{1},...,\alpha_{K}=\frac{1}{K}` 

Expectation (E) Step
````````````````````````````
计算数据点 :math:`x_i` 由子模型 :math:`C_k` 生成的期望   

.. math:: 
    \gamma_{ik}=\frac{\alpha_k\eta(x_i|\mu_k,\sigma_k)}{\sum_{j=1}^{K}\alpha_j\eta(x_i|\mu_j,\sigma_j)}


其中 :math:`\gamma_{ik}` 表示 :math:`x_i` 由第k个子模型 :math:`C_k` 生成的概率,所以 :math:`\gamma_{ik}=p(C_k|x_i,\alpha_k,\mu_k,\sigma_k)`     


Maximization (M) Step
`````````````````````````````````````
.. math:: 
  \begin{eqnarray*}
  && \alpha_k = \sum_{i=1}^{N}\frac{\gamma_{ik}}{N} \\
  && \mu_k = \frac{\sum_{i=1}^{N}\gamma_{ik}x_i}{\sum_{i=1}^{N}\gamma_{ik}} \\
  && \sigma_k = \frac{\sum_{i=1}^{N}\gamma_{ik}(x_i-\mu_k)^2}{\sum_{i=1}^{N}\gamma_{ik}}
  \end{eqnarray*}

推理
``````````````
数据x属于子模型 :math:`C_i` 的概率：

.. math:: 
    P(C_i|x) = \frac{P(x,C_i)}{P(x)} = \frac{P(C_i)P(x|C_i)}{\sum_{j=1}^{K}P(C_j)P(x|C_j)} = \frac{\alpha_i\eta(x|\mu_i,\sigma_i)}{\sum_{j=1}^{K}\alpha_j\eta(x|\mu_j,\sigma_j)}


参考
---------
https://brilliant.org/wiki/gaussian-mixture-model/              
https://zhuanlan.zhihu.com/p/30483076   
https://www.geeksforgeeks.org/gaussian-mixture-model/
