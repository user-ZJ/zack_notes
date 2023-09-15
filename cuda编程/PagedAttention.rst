.. _PagedAttention:

PagedAttention
============================
vLLM 主要用于快速 LLM 推理和服务，其核心是 PagedAttention，这是一种新颖的注意力算法，它将在操作系统的虚拟内存中分页的经典思想引入到 LLM 服务中。
在无需任何模型架构修改的情况下，可以做到比 HuggingFace Transformers 提供高达 24 倍的 Throughput。

PagedAttention 的基本原理
----------------------------------

PagedAttention：如何解决 GPU 显存瓶颈
```````````````````````````````````````````````
LM 服务的性能受到内存瓶颈的影响。在自回归 decoder 中，所有输入到 LLM 的 token 会产生注意力 key 和 value 的张量，这些张量保存在 GPU 显存中以生成下一个 token。
这些缓存 key 和 value 的张量通常被称为 KV cache，其具有以下特点：

1. 显存占用大：在 LLaMA-13B 中，缓存单个序列最多需要 1.7GB 显存
2. 动态变化：KV 缓存的大小取决于序列长度，这是长度可变和不可预测的。因此，这对有效管理 KV cache 挑战较大。该研究发现，由于碎片化和过度保留，现有系统浪费了 60% - 80% 的显存。

为了解决这个问题，该研究引入了 PagedAttention，这是一种受操作系统中虚拟内存和分页经典思想启发的注意力算法。
与传统的注意力算法不同，PagedAttention 允许在非连续的内存空间中存储连续的 key 和 value 。
具体来说，PagedAttention 将每个序列的 KV cache 划分为块，每个块包含固定数量 token 的键和值。在注意力计算期间，PagedAttention 内核可以有效地识别和获取这些块。

.. image:: /images/cuda编程/PagedAttention1.webp

因为块在内存中不需要连续，因而可以用一种更加灵活的方式管理 key 和 value ，就像在操作系统的虚拟内存中一样：
可以将块视为页面，将 token 视为字节，将序列视为进程。序列的连续逻辑块通过块表映射到非连续物理块中。物理块在生成新 token 时按需分配。

.. image:: /images/cuda编程/PagedAttention2.webp

在 PagedAttention 中，内存浪费只会发生在序列的最后一个块中。这使得在实践中可以实现接近最佳的内存使用，仅浪费不到 4%。
这种内存效率的提升被证明非常有用，允许系统将更多序列进行批处理，提高 GPU 使用率，显著提升吞吐量。

PagedAttention 还有另一个关键优势 —— 高效的内存共享。例如在并行采样中，多个输出序列是由同一个 prompt 生成的。在这种情况下，prompt 的计算和内存可以在输出序列中共享。

.. image:: /images/cuda编程/PagedAttention3.webp

PagedAttention 自然地通过其块表格来启动内存共享。与进程共享物理页面的方式类似，PagedAttention 中的不同序列可以通过将它们的逻辑块映射到同一个物理块的方式来共享块。
为了确保安全共享，PagedAttention 会对物理块的引用计数进行跟踪，并实现写时复制（Copy-on-Write）机制。


.. image:: /images/cuda编程/PagedAttention4.webp

PageAttention 的内存共享大大减少了复杂采样算法的内存开销，例如并行采样和集束搜索的内存使用量降低了 55%。这可以转化为高达 2.2 倍的吞吐量提升。这种采样方法也在 LLM 服务中变得实用起来。


