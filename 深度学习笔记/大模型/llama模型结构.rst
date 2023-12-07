llama模型结构
===============================


.. figure:: /images/深度学习/llama2.png
    :align: center


:download:`大模型结构.pptx </images/resources/大模型结构图.pptx>`


kv_cache: 

13B模型  1x40xtx128

7B模型   1x32xtx128


kv_cache作用计算过程
--------------------------------
以llama7b模型为例，在t时刻，计算出来的QKV维度均为：1x32x1x128；

此时kv_cache维度均为1x32x(t-1)x128,和当前的kv合并后维度为1x32xtx128；

Q和K的转置计算后进行矩阵乘的结果attn_weights维度为1x32x1xt；

attn_weights和V矩阵乘的结果attn_output维度为1x32x1x128


https://zhuanlan.zhihu.com/p/649756898

https://www.bilibili.com/video/BV12h4y1N7C8/?vd_source=1cbcdbb91c2e108ff4f290eeb865ee30


