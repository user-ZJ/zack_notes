# Packed-Input Batching 完整介绍

## 概述

Packed-Input Batching（打包输入批处理）是一类面向变长序列的训练与推理优化技术。它的核心目标是：**不再把所有样本补齐到批内最长长度，而是只存储和计算有效 token，并用额外的边界元数据保持各样本在逻辑上相互独立。**

对 Transformer 训练而言，最典型的实现是将 batch 维和 sequence 维折叠成连续的 token 维，再通过 `cu_seqlens` 告诉变长注意力内核每条序列的起止位置：

```text
传统 padded batch:  [B, S_max, D]
打包后的 THD 布局: [T, H, D]，其中 T = Σ s_i
边界元数据:        cu_seqlens = [0, s_0, s_0+s_1, ..., T]
```

这一思想看似只是“把短序列拼起来”，但正确实现还必须同时解决四个问题：

1. **边界隔离**：不同样本不能互相注意，也不能在卷积或状态递推中互相污染。
2. **内核支持**：算子必须能够直接处理变长、无 padding 的布局；仅仅拼接张量并不会自动节省计算。
3. **损失归一化**：不同 micro-batch 的有效 token 数变化后，loss 不能继续按固定 batch 大小简单平均。
4. **调度与负载均衡**：打包消除了 padding，但不同序列的二次方注意力开销仍可能造成 GPU 负载不均。

本文系统介绍 Packed-Input Batching 的概念边界、数据布局、核心元数据、正确性要求、主流实现路线、LLM 推理中的相关技术，以及语音和 Conformer 模型中的特殊难点。

---

## 一、为什么需要 Packed-Input Batching

### 1.1 传统 padding 的计算浪费

常规 mini-batch 要求张量具有规则形状。若一个批次包含 8 条序列，长度为：

```text
[4, 6, 8, 5, 4, 3, 7, 8]
```

将它们全部补齐到长度 8 后，需要保存和处理：

```text
8 × 8 = 64 token
```

但真正有效的 token 只有：

```text
4 + 6 + 8 + 5 + 4 + 3 + 7 + 8 = 45 token
```

因此该批次有 19 个 padding token，占总输入的约 29.7%。这些 token 没有建模价值，却可能经过线性层、归一化层、MLP 和注意力层，并占用激活显存。

设第 $i$ 条序列长度为 $s_i$，batch 大小为 $B$，批内最大长度为 $S_{\max}$，则 token 层面的 padding 利用率为：

$$
\eta_{\text{token}} = \frac{\sum_{i=1}^{B}s_i}{B S_{\max}}
$$

padding 浪费率为：

$$
r_{\text{padding}} = 1-\eta_{\text{token}}
$$

序列长度方差越大、batch 越大，浪费通常越严重。

### 1.2 注意力中的额外浪费

对标准全局自注意力，padded batch 的主要计算量近似为：

$$
O(B S_{\max}^{2})
$$

若变长内核按各序列真实长度分别计算，则主要计算量变为：

$$
O\left(\sum_{i=1}^{B}s_i^{2}\right)
$$

这里必须注意：**只有支持变长或块稀疏计算的内核，才能真正得到后一种复杂度。** 如果只是把全部序列拼成总长度 $T=\sum_i s_i$ 的长序列，再物化一个普通稠密掩码，计算量可能接近 $O(T^2)$，反而比 padding 更差。

### 1.3 收益没有统一常数

Packed batching 的收益取决于数据长度分布、模型结构、attention 占比、硬件和内核实现。公开测试中的典型结果包括：

| 场景 | 公开结果 | 说明 |
|---|---:|---|
| FLAN，8×A100 | 吞吐约 2×，峰值显存约降低 20% | 短序列多、长度方差大 |
| OrcaMath，8×A100 | 吞吐约 1.4×，峰值显存约降低 6% | 序列较长且长度更均匀 |
| ESM-2 蛋白模型，上下文 1024 | padding token 占比可达约 60% | NVIDIA BioNeMo 报告场景 |

这些数字来自特定配置，不能直接视为所有模型的通用收益。判断一个任务是否适合打包，应先统计长度直方图、padding 比例和 attention 的运行时间占比。

---

## 二、容易混淆的概念

“Packing”在不同框架和论文中可能指不同技术。讨论前应先区分以下概念。

### 2.1 拼接打包（concatenation packing）

将多条短样本拼到一个固定长度容器中，以提高容器填充率。例如：

```text
[A A A] + [B B] + [C C C C]
             ↓
[A A A B B C C C C]
```

拼接本身只改变数据组织方式。若注意力和其他时序算子不知道样本边界，B 可以读取 A，C 可以读取 A 和 B，从而产生交叉污染。

### 2.2 扁平化或无 padding 训练（flattening / padding-free training）

把 mini-batch 中所有有效 token 连续放置，通常得到 `[1, T]` 的 token 输入或 `[T, H, D]` 的注意力输入，同时提供边界元数据。Hugging Face `DataCollatorWithFlattening` 属于这一类。

它强调的是“去掉 padding”，不一定要求把样本填入固定长度的 bin。

### 2.3 变长内核（varlen kernel）

变长内核接收连续 token 张量和序列边界，只对各段内部执行计算。FlashAttention varlen、Transformer Engine 的 THD attention，以及 PyTorch 原生 `varlen_attn` 都属于这一层。

**数据打包负责形成布局，变长内核负责兑现计算收益；两者缺一不可。**

### 2.4 分桶与动态批处理（bucketing / dynamic batching）

分桶把长度相近的样本放入同一批次；动态批处理则按 token 数、帧数或音频总时长决定每批包含多少样本。它们不一定消除 padding，但能显著减少 padding，并改善负载均衡。

分桶与打包不是互斥方案。实际系统经常先分桶，再在桶内打包。

### 2.5 LLM 推理中的连续批处理（continuous batching）

Continuous batching 是推理服务的调度技术：请求完成后立即移出，新请求可在下一次迭代加入，不必等待整个静态 batch 结束。

它与训练侧 packed-input batching 都在处理变长序列，也可能使用连续 token 布局，但二者并不等价：

| 维度 | 训练侧 sequence packing | 推理侧 continuous batching |
|---|---|---|
| 主要目标 | 消除 padding 与无效激活 | 提高在线吞吐和 GPU 占用 |
| 状态 | 通常一次前向处理完整训练样本 | 请求跨多个 decode step 持续存在 |
| 关键元数据 | `cu_seqlens`、`max_seqlen`、`seq_idx` | 请求表、slot、KV block table、序列长度 |
| 核心难点 | 边界隔离、loss 归一化 | 动态调度、KV cache 分配、延迟控制 |

### 2.6 PagedAttention

PagedAttention 主要解决推理 KV cache 的分页存储、碎片和共享问题。它常与 continuous batching 一起出现，但不是训练侧 sequence packing 的同义词，也不是 FlashAttention varlen 的替代名称。

---

## 三、核心数据表示

### 3.1 一个贯穿全文的例子

设有三条序列：

```text
A = [a0, a1, a2]          长度 3
B = [b0, b1]              长度 2
C = [c0, c1, c2, c3]      长度 4
```

传统 padded batch 为：

```text
[
  [a0, a1, a2, PAD],
  [b0, b1, PAD, PAD],
  [c0, c1, c2, c3]
]
```

形状为 `[3, 4]`，共有 12 个位置，其中只有 9 个有效。

打包后为：

```text
[a0, a1, a2, b0, b1, c0, c1, c2, c3]
```

总 token 数：

```text
T = 3 + 2 + 4 = 9
```

对应累计序列长度：

```text
cu_seqlens = [0, 3, 5, 9]
max_seqlen = 4
```

第 $i$ 条序列在扁平张量中的范围是：

```text
[cu_seqlens[i], cu_seqlens[i + 1])
```

例如 B 的范围是 `[3, 5)`。

### 3.2 `cu_seqlens` 的准确语义

`cu_seqlens` 是 cumulative sequence lengths，即序列长度的前缀和。它有以下硬性或常见约束：

- 长度为 `B + 1`；
- 第一个元素恒为 0；
- 最后一个元素等于总 token 数 `T`；
- 相邻元素之差是对应序列长度；
- FlashAttention 等常见 CUDA 接口要求 dtype 为 `torch.int32`；
- 序列边界必须单调递增。

构造方式：

```python
import torch
import torch.nn.functional as F

lengths = torch.tensor([3, 2, 4], device="cuda")
cu_seqlens = F.pad(
    lengths.cumsum(dim=0, dtype=torch.int32),
    (1, 0),
)
# tensor([0, 3, 5, 9], dtype=torch.int32)
```

使用前缀和而不是普通长度数组，是因为内核需要快速取得每一段的基地址。`cu_seqlens[i]` 可以在 O(1) 时间给出第 $i$ 段的起点。

### 3.3 `max_seqlen`

`max_seqlen` 是批内最长真实序列长度。本例中为 4。它通常用于 kernel launch、tile 数量和共享内存配置。

在 Transformer Engine 的 THD 路径中，如果不显式传入最大长度，某些实现会从设备张量推断它，从而引入 GPU 到 CPU 的拷贝和同步。工程上应尽量在 collator 或调度器中预先计算并显式传递。

### 3.4 `position_ids`

打包后每条样本的位置通常需要重新起算：

```text
position_ids = [0, 1, 2, 0, 1, 0, 1, 2, 3]
```

这既保持每条样本原有的位置语义，也可在部分标准注意力模型中辅助推断边界。但它不能替代通用边界元数据：

- Hugging Face 的运行时边界推断通常只适用于扁平后 `batch_size == 1` 的标准注意力模型；
- 数据相关的边界检查可能造成 `torch.compile` graph break；
- 卷积、Gated DeltaNet、其他线性注意力和 SSM 扫描通常不会从 `position_ids` 获取边界；
- 某些模型的位置编号不从 0 开始，例如 RoBERTa 系位置 embedding 与 `padding_idx` 有特定约定。

因此更稳妥的做法是由 data collator 预计算边界，而不是依赖模型在前向过程中从 `position_ids` 猜测。

### 3.5 `seq_idx`

`seq_idx` 为每个 token 标记所属样本：

```text
seq_idx = [0, 0, 0, 1, 1, 2, 2, 2, 2]
```

它适合表达卷积、线性注意力、递推扫描等算子的段边界。若模型中含有这些组件，仅提供 `cu_seqlens` 给自注意力层并不足以保证整个模型正确。

### 3.6 BSHD、SBHD 与 THD

常见注意力布局如下：

| 布局 | 形状 | 含义 |
|---|---|---|
| `bshd` | `[B, S, H, D]` | batch-first 的常规 padded 布局 |
| `sbhd` | `[S, B, H, D]` | sequence-first 布局，Megatron 中较常见 |
| `thd` | `[T, H, D]` | 将 batch 与 sequence 合并后的变长布局 |

其中：

- `H` 为注意力头数；
- `D` 为每头维度；
- $T = \sum_i s_i$ 为所有有效 token 的总数。

华为昇腾生态中常见的 `TND` 与 THD 同构，其中 `N` 通常表示注意力头数，不应解释为序列数量。

---

## 四、边界隔离与交叉污染

### 4.1 为什么普通 causal mask 不够

假设把 A 和 B 直接拼接：

```text
[a0, a1, a2, b0, b1]
```

普通 causal mask 只禁止 token 查看未来位置。对 `b0` 而言，A 位于过去，因此仍然可见：

```text
b0 可见 [a0, a1, a2, b0]
b1 可见 [a0, a1, a2, b0, b1]
```

正确的 packed causal attention 应形成块对角下三角结构：

```text
      A0 A1 A2 B0 B1
A0     ✓
A1     ✓  ✓
A2     ✓  ✓  ✓
B0              ✓
B1              ✓  ✓
```

不同样本之间必须完全不可见。

### 4.2 交叉污染的后果

若边界未隔离：

- 后一条样本获得前一条样本的额外上下文，训练 loss 可能虚假降低；
- 推理时不存在这些上下文，训练和推理分布不一致；
- eval 指标可能下降；
- 不同用户或文档之间的信息流动还可能构成隐私风险。

修复污染后训练 loss 变高并不一定是退化，因为模型失去了原本“白送”的跨样本信息。应以 WER、CER、准确率、困惑度等独立验证指标判断效果。

### 4.3 非注意力算子的污染

块对角 attention 只保护注意力层。下列算子同样沿时间维聚合信息：

- causal convolution；
- Conformer depthwise convolution；
- Conv1d/Conv2d 下采样；
- Gated DeltaNet 等线性注意力；
- Mamba 等状态空间模型的 scan；
- FFT 长卷积。

这些算子若不知道边界，会把样本 A 的尾部状态或卷积窗口传播到样本 B 的头部。更危险的是，很多参考实现会静默把整个扁平 batch 当成一条长序列，而不报错。

RubiConv 的研究进一步指出，连续长卷积存在相邻文档边界渗漏，FFT 实现还可能引入序列首尾的周期环绕污染。

### 4.4 验证边界隔离的方法

工程上应至少做以下一致性测试：

1. 将相同样本分别以 padded 和 packed 方式前向，关闭 dropout 后比较有效位置输出；
2. 固定样本 B，只替换打包在它前面的样本 A，确认 B 的输出不变；
3. 比较 packed 与 unpacked 的梯度；
4. 对 conv/SSM 模型分别测试样本头尾位置；
5. 通过极小输入手工检查 attention mask 或 `seq_idx`；
6. 在不同 batch 排列顺序下比较 eval 指标。

第二项是发现跨样本污染尤其有效的单元测试。

---

## 五、RNN 中的 PackedSequence

Packed sequence 最早在工程中广泛用于 RNN、LSTM 和 GRU。PyTorch 提供：

- `pack_padded_sequence`：把 padded batch 转换为 `PackedSequence`；
- `pack_sequence`：直接打包张量列表；
- `pad_packed_sequence`：将输出恢复为 padded 张量。

### 5.1 基本示例

```python
import torch
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

x = torch.randn(3, 4, 80)      # [B, S_max, D]
lengths = torch.tensor([3, 2, 4])

packed = pack_padded_sequence(
    x,
    lengths.cpu(),
    batch_first=True,
    enforce_sorted=False,
)

lstm = torch.nn.LSTM(80, 256, batch_first=True)
packed_out, (h_n, c_n) = lstm(packed)

out, restored_lengths = pad_packed_sequence(
    packed_out,
    batch_first=True,
)
```

### 5.2 内部表示

PyTorch `PackedSequence` 的核心字段包括：

- `data`：按有效时间步重排后的连续数据；
- `batch_sizes`：每个时间步仍然活跃的序列数；
- `sorted_indices`：原始 batch 到排序后 batch 的映射；
- `unsorted_indices`：恢复原顺序的映射。

这与 Transformer 常用的 THD + `cu_seqlens` 不是同一种内存组织方式。

### 5.3 排序要求的准确说法

`pack_padded_sequence` 的 `enforce_sorted` 默认值为 `True`，此时输入必须按长度降序排列；设置 `enforce_sorted=False` 后，PyTorch 可在内部完成排序。

因此“PyTorch PackedSequence 强制要求用户预排序”并不准确。某些导出路径或旧版本工具链可能仍对排序有额外要求，应按实际版本验证。

### 5.4 与 Transformer packing 的区别

RNN 每个时间步递推，`batch_sizes` 可以让已结束的序列退出计算。Transformer 则需要处理序列内任意位置间的注意力关系，必须使用块对角 mask 或变长注意力内核。推理时 Transformer 还要管理 KV cache，这也是 RNN PackedSequence 没有覆盖的问题。

---

## 六、Transformer 中的实现原理

### 6.1 数据侧流程

一个完整的训练侧打包流程通常包括：

1. 读取样本并计算有效长度；
2. 可选地按长度分桶；
3. 使用 FFD、BFD 等算法组合样本；
4. 拼接 `input_ids`、`labels` 和其他逐 token 特征；
5. 重置或按模型规则生成 `position_ids`；
6. 构造 `cu_seqlens`、`max_seqlen`，必要时构造 `seq_idx`；
7. 在模型中使用 varlen attention 或块稀疏 attention；
8. 按有效预测 token 总数归一化 loss；
9. 必要时将输出解包回原始样本结构。

### 6.2 内核侧流程

变长 attention 接收：

```text
Q: [T_q, H_q, D]
K: [T_k, H_kv, D]
V: [T_k, H_kv, D]
cu_seqlens_q:  [B + 1]
cu_seqlens_kv: [B + 1]
max_seqlen_q
max_seqlen_kv
```

内核根据累计长度为每条序列定位 Q/K/V 的连续区间，只在匹配的序列段内部计算 attention。

“内核一定通过二分查找 `cu_seqlens` 定位序列”并不是通用事实。不同 FlashAttention 版本和调度器可能直接使用 batch grid、前缀和、投票、除法映射或可选的二分提示。对使用者而言，稳定的接口契约是边界数组和最大长度，而不是某一种内部查找算法。

### 6.3 FlashAttention varlen 示例

```python
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,                              # [T, Hq, D]
    k,                              # [T, Hkv, D]
    v,                              # [T, Hkv, D]
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
```

对 self-attention，还可使用 QKV packed 版本：

```python
from flash_attn import flash_attn_varlen_qkvpacked_func

output = flash_attn_varlen_qkvpacked_func(
    qkv,             # [T, 3, H, D]
    cu_seqlens,
    max_seqlen,
    causal=True,
)
```

若 Q 的头数与 K/V 不同，即 GQA 或 MQA，应选择支持独立 Q/K/V 或 KV-packed 的接口，并满足头数整除要求。

### 6.4 Hugging Face padding-free 路线

```python
from transformers import (
    AutoModelForCausalLM,
    DataCollatorWithFlattening,
    Trainer,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
)

data_collator = DataCollatorWithFlattening(
    return_flash_attn_kwargs=True,
    return_seq_idx=True,  # 模型含 conv/线性注意力/SSM 时尤其重要
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
```

关键点：

- collator 通常把输入变为 `[1, T]`；
- `return_flash_attn_kwargs=True` 预计算边界和最大长度，避免运行时从 `position_ids` 推断；
- `return_seq_idx=True` 为需要样本 ID 的非标准时序算子提供边界；
- 不应再传与扁平布局冲突的传统二维 `attention_mask`；
- `separator_id=-100` 等参数可用于确保样本边界处不产生错误的语言模型预测目标；
- `position_ids_start` 应与模型预训练时的位置编号规则一致。

只有模型及所选 attention backend 真正接收这些参数时，collator 的输出才会生效。集成后应检查 forward signature、Trainer 保留列和内核回退路径，避免元数据被静默丢弃。

### 6.5 FlexAttention block mask 路线

当注意力规则不只是“每条样本独立的全局/因果注意力”时，可以使用 PyTorch FlexAttention 的 `BlockMask` 表达组合约束，例如：

- packed causal attention；
- 文档边界 + 滑动窗口；
- 文档边界 + prefix-LM；
- 语音样本边界 + streaming chunk mask。

其优势是掩码表达力强，且可跳过无效块；代价是需要编译，存在冷启动开销，性能也可能低于针对简单 varlen 场景高度优化的 FlashAttention。

### 6.6 Transformer Engine 与 Megatron THD 路线

在大规模训练中，Megatron-LM 和 Transformer Engine 使用 `PackedSeqParams` 传递 THD 所需元数据：

```python
PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=cu_q,
    cu_seqlens_kv=cu_kv,
    max_seqlen_q=max_q,
    max_seqlen_kv=max_kv,
)
```

工程配置应区分两类机制：

- `reset-position-ids`、`reset-attention-mask` 等传统文档级 masking，主要解决定长块中的文档边界；
- `is_packed_sequence=True`、varlen dataset 和 THD attention，才是现代 padding-free 路径。

若 dataloader 已提供累计长度，应避免再次物化巨大的 `[B, 1, S, S]` attention mask。

### 6.7 PyTorch 原生 varlen attention

PyTorch 新版提供 `torch.nn.attention.varlen.varlen_attn`，并将其设计为可被 `torch.compile` 处理的 custom op。其接口同样接收 Q/K/V、累计长度和最大长度，并通过 `window_size` 表达双向、因果或滑动窗口注意力。

这条路线减少了项目对第三方 FlashAttention 编译扩展的直接依赖，但硬件、dtype、PyTorch 版本和后端覆盖仍需按部署环境核实。

---

## 七、Loss、标签与位置编码

### 7.1 样本边界处的标签

对 causal language modeling，样本 A 的最后一个 token 不应以样本 B 的第一个 token 作为 next-token label。通常有两种做法：

- 在样本间插入 EOS，并让预测 EOS 成为 A 的最后一个训练目标；
- 将跨边界位置的 label 设为 `-100`，使交叉熵忽略该位置。

边界隔离 attention 并不会自动修正 labels，数据整理器必须显式处理。

### 7.2 梯度累积中的 loss 归一化

打包后每个 micro-batch 的有效预测 token 数可能不同。若先对每个 micro-batch 独立取 mean，再等权累加，短 batch 与长 batch 会获得相同权重，结果不等价于在完整 batch 上计算 token 平均 loss。

更稳妥的原则是：

```python
loss_sum = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1),
    ignore_index=-100,
    reduction="sum",
)

loss = loss_sum / num_valid_target_tokens
```

采用梯度累积或多数据并行 rank 时，分母应与框架的梯度缩放和 all-reduce 语义一致，通常需要使用整个累积窗口内的有效目标总数。

### 7.3 Token 等权与样本等权

扁平后直接对所有 token 求均值，会使长样本权重更高。例如 2000-token 样本对目标函数的贡献约是 100-token 样本的 20 倍。

这不是实现错误，而是建模口径：

- **token 等权**：适合大多数语言模型训练；
- **样本等权**：先对每条样本内部平均，再对样本平均；
- **任务加权**：多任务或多模态训练中按数据源、任务或模态分配权重。

采用 packing 前后应保证目标函数口径符合预期。

### 7.4 位置编码兼容性

打包时通常需要按样本重置位置，但不能机械地从 0 开始。应检查：

- learned absolute position embedding 的起始偏移；
- RoPE 的位置重置规则；
- 多模态 3D position IDs 的形状；
- sliding-window attention 的局部位置约定；
- checkpoint 预训练阶段是否跨文档连续计数。

sentence-transformers 5.6.1 记录过 RoBERTa 类模型位置起点不匹配导致指标大幅下降的案例，说明“输出形状正确”不代表位置语义正确。

---

## 八、打包算法与负载均衡

### 8.1 常见打包算法

设每个容器容量为 $L$，每条序列长度为 $s_i$。打包问题可视为 bin packing：

| 算法 | 典型复杂度 | 特点 |
|---|---:|---|
| First Fit Decreasing（FFD） | $O(n\log n)$ | 长度降序后放入第一个可容纳的 bin，工程中常用 |
| Best Fit Decreasing（BFD） | $O(n\log n)$ | 放入剩余空间最小但仍可容纳的 bin，填充率通常较高 |
| 贪心 knapsack | 依实现而定 | 可针对固定 token 容量优化 |
| 定长硬切 | $O(n)$ | 简单，但可能截断或混合样本语义 |

`wrapped` 一类策略允许样本跨容器边界切分，虽然能提高利用率，却可能混合无关样本或改变训练样本定义，应谨慎使用。

### 8.2 为什么打包后仍要分桶

两个 pack 即使总 token 数相同，attention 计算量也可能相差很大：

```text
Pack 1: [1000, 1000]  → 1000² + 1000² = 2,000,000
Pack 2: [1900, 100]   → 1900² + 100² = 3,620,000
```

因此只平衡总 token 数并不能保证计算量相同。长度分桶、按平方代价估算、动态调度和 sequence packing scheduler 仍有价值。

### 8.3 与并行策略的关系

| 并行方式 | 与 packing 的关系 | 关键注意点 |
|---|---|---|
| Tensor Parallel（TP） | 通常与 THD 正交，沿 head/hidden 维切分 | token 对齐和通信融合仍可能有约束 |
| Sequence Parallel（SP） | 可沿 THD 的 T 维切分逐 token 算子 | 对齐 padding 可能影响后端选择 |
| Context Parallel（CP） | 最复杂，可能在样本内部切序列 | 不能简单均分 T 后丢失文档边界 |
| Pipeline Parallel（PP） | 各 stage 都需要一致的边界元数据 | 需传递或广播最大长度和累计长度 |
| Data Parallel（DP） | 各 rank 的序列长度组成不同 | 应平衡 $\sum_i s_i^2$ 等实际负载 |

### 8.4 Context Parallel 的特殊约束

CP 会把长上下文分给多个 rank。如果直接等分 THD 的 T 维，切点可能落在样本内部，原始 `cu_seqlens` 不能再直接描述每个 rank 的局部布局。

Megatron/Transformer Engine 会使用按文档或按序列的分区索引，在序列内部采用特定平衡切法，并保留必要元数据。某些实现还要求：

- 全局序列长度可被 `2 × cp_size` 整除；
- 每条独立序列为 CP 对齐补齐；
- 选择与拓扑匹配的 P2P、all-gather 或 all-to-all 通信。

这些约束与版本紧密相关，不能仅凭普通 FlashAttention 的单卡接口推断多卡行为。

---

## 九、LLM 推理：Continuous Batching 与 KV Cache

### 9.1 Prefill 与 Decode 的差异

LLM 推理包含两个主要阶段：

- **Prefill**：一次处理整段 prompt，输入 token 较多，矩阵乘法规模大，偏计算密集；
- **Decode**：每个请求每轮通常只生成一个 token，同时读取历史 KV cache，偏内存带宽密集。

请求的 prompt 长度和生成长度不同，因此静态 batch 会产生严重的等待和空槽。

### 9.2 静态批处理

静态批处理在开始时选定一组请求，通常等所有请求完成后才释放整个 batch。较短请求提前结束后，其计算槽位可能闲置，直到最长请求结束。

### 9.3 迭代级连续批处理

Continuous batching 在每个调度迭代重新决定活跃请求：

1. 已完成的请求立即退出；
2. 新请求进入空出的容量；
3. 调度器组合 decode token 和可接受的 prefill token；
4. 根据 token 预算、KV cache 容量和延迟目标发起下一轮计算。

因此 batch 的成员和形状会持续变化。

### 9.4 Chunked Prefill

长 prompt 的 prefill 可能长时间占据 GPU，阻塞 decode 请求并增加在线延迟。Chunked prefill 将长 prompt 切成多个 token 块，在多个调度轮中逐步处理，从而让 prefill 与 decode 更公平地共享预算。

### 9.5 KV Cache 管理

每个活跃请求都需要保存历史 K/V。若按最大长度为每个请求预分配连续空间，会造成浪费和外部碎片。PagedAttention 将 KV cache 切成固定大小的 block，并用 block table 把逻辑 token 位置映射到物理块：

```text
逻辑序列 token
      ↓
请求 block table
      ↓
分散的物理 KV blocks
```

这种方式支持：

- 按需增长 KV cache；
- 请求完成后快速回收 block；
- 减少连续大块分配造成的碎片；
- beam search、并行采样或共享前缀中的块共享。

### 9.6 与训练侧 packing 的联系和边界

推理引擎也常把当前迭代的有效 token 拼成连续数组，并通过元数据描述请求归属，这与 THD 思想相似。但推理还必须保存跨迭代状态、处理动态请求和管理物理 KV block，因此不能把 continuous batching 简化为“训练 packing 加一个 KV cache”。

TensorRT-LLM、vLLM 等框架对上下文阶段、生成阶段、混合批次和 KV cache 的具体调度约束也不同，使用时应以对应版本文档为准。

---

## 十、卷积、线性注意力与 SSM

### 10.1 为什么 `position_ids` 路径无效

标准自注意力可以通过显式边界或某些位置归零点推断样本段。但卷积核只读取相邻内存位置，SSM scan 只递推前一状态；它们通常根本不读取 `position_ids`。

因此：

```text
重置 position_ids ≠ 重置卷积感受野
重置 position_ids ≠ 重置 SSM hidden state
```

必须由 `seq_idx`、segment pointer、分段 kernel 或显式状态重置机制处理。

### 10.2 可选解决路线

| 路线 | 做法 | 优点 | 代价 |
|---|---|---|---|
| 分层打包 | 先在 padded 域完成 conv，再 unpad 进入 Transformer | 实现风险低 | 前端 padding 仍存在 |
| 边界间隔 | 样本间插入至少覆盖卷积感受野的静音/零帧 | 无需改 kernel | 仍有少量无效计算，边界语义需验证 |
| 分段卷积 | 按样本切出后独立卷积 | 语义严格正确 | 小序列循环可能降低并行度 |
| 边界感知 kernel | kernel 根据 `seq_idx` 屏蔽跨段读取或重置状态 | 正确且高效 | 开发与维护成本高 |
| 输出修正/重算 | 对边界附近受污染位置单独处理 | 可渐进改造 | 实现复杂，容易漏掉梯度或多层感受野 |

间隔长度不能只看单层 `kernel_size - 1`。多层卷积、dilation 和 stride 会扩大有效感受野，应按整个前端计算。

---

## 十一、语音与 Conformer 场景

### 11.1 语音为何更需要变长优化

语音样本从几秒到数小时不等，帧长度方差通常比文本 token 长度更大。将所有样本 pad 到最长音频会浪费大量声学前端、下采样和编码器计算。

因此主流语音框架首先采用按时长或帧数动态组批：

| 框架 | 典型机制 |
|---|---|
| ESPnet2 | `length`、`numel` 等 batch type，使用 `batch_bins` 控制批容量 |
| Lhotse / icefall | `DynamicBucketingSampler`，按 `max_duration` 控制总音频时长 |
| NeMo | tarred dataset bucketing、semi-sorted batching |
| SpeechBrain | `DynamicBatchSampler` 与长度桶 |
| FunASR | 按批内总音频秒数设置 batch |

这些方案是低风险的第一步，即使不修改模型内核也常能获得明显收益。

### 11.2 Conformer 的特殊困难

Conformer 同时包含：

1. Conv2d subsampling 前端；
2. 自注意力模块；
3. 每个 Conformer block 内的 depthwise convolution；
4. 流式模型中的 chunk mask 和缓存。

FlashAttention varlen 只能直接解决第 2 项。若把多条音频帧从输入端直接拼接：

- subsampling 卷积会跨样本读取；
- Conformer convolution module 会在边界混合局部特征；
- streaming cache 可能把前一条样本状态带入后一条；
- 单一 `cu_seqlens` 不能完整表达 chunk 级可见性。

### 11.3 推荐的分层打包方案

对“Conv 前端 + Conformer/Transformer + LLM”的模型，较务实的渐进路线是：

```text
变长音频
  ↓
按长度分桶并 padded
  ↓
Conv/Subsampling（保持常规 batch）
  ↓
根据下采样后真实长度 unpad
  ↓
Conformer attention 或纯 Transformer 层使用 THD/varlen
  ↓
需要卷积的 Conformer block 使用边界感知方案
  ↓
LLM decoder 侧独立实施 padding-free packing
```

第一阶段可以只 pack LLM decoder，音频塔保持 padded；第二阶段在 subsampling 后 unpad；第三阶段再改造 Conformer conv module。这样能将收益和正确性风险分开验证。

### 11.4 流式 ASR

流式 Conformer 的可见区域通常是带状或分块结构，需要同时满足：

- token 属于同一条音频；
- token 位于允许访问的历史 chunk、当前 chunk 或有限前瞻范围。

普通 varlen causal attention 只能表达“同一样本 + 因果”这一简单规则。复杂 chunk mask 更适合用 FlexAttention `mask_mod` 或专用 streaming kernel 表达。

### 11.5 Whisper 固定窗口

原版 Whisper 将音频 pad 或裁剪到固定 30 秒窗口，短音频中的 padding 仍会经过编码器。可采用 VAD 切分、按时长动态批或在严格边界隔离下把多段短音频装入固定窗口。

但直接把多段 mel 特征拼入 3000 帧而只加 attention mask 并不充分，因为卷积前端仍可能跨段混合。

### 11.6 语音模型验证指标

语音 packing 实验应至少比较：

- WER / CER；
- 相同随机种子下的收敛速度；
- 有效音频秒数/秒与有效帧数/秒；
- 峰值显存；
- 不同音频排序下的输出一致性；
- 边界附近帧的隐藏状态差异；
- 流式首包延迟与实时率。

训练 loss 只能作为辅助指标，不能代替 WER/CER。

---

## 十二、多模态场景

多模态模型通常同时处理文本 token、图像 patch 和音频帧。Packing 需要回答三个问题：

1. 各模态是否共享一个 token 轴；
2. 每种模态的前端是否支持变长和边界隔离；
3. 跨模态 attention 哪些区域应当可见。

可行的工程分层通常是：

- 图像编码器按分辨率分桶或保持 padded；
- 音频卷积前端按时长动态批；
- 编码后的视觉/音频 token 与文本 token 在 LLM 侧统一打包；
- 用 segment ID 同时表达样本归属和模态归属；
- 在 labels 中只保留需要监督的模态位置。

多模态 position IDs 可能是二维或三维结构，不能套用“扫描 0 位置即可推断序列边界”的简单逻辑。曾有 FlashAttention 集成因 3D M-RoPE `position_ids` 被误判为 packed metadata 而触发非法内存访问的案例。

---

## 十三、常见错误与排查

### 13.1 只拼接，不隔离

**现象**：训练 loss 异常好，但 eval 下降；改变样本排列顺序会改变输出。

**原因**：普通 causal mask 允许后一条样本看到前一条样本。

**处理**：使用 varlen attention 的显式 `cu_seqlens`，或正确的块对角 mask。

### 13.2 元数据 dtype 或形状错误

应检查：

```text
cu_seqlens.dtype == torch.int32
len(cu_seqlens) == batch_size + 1
cu_seqlens[0] == 0
cu_seqlens[-1] == total_tokens
cu_seqlens 单调递增
max_seqlen == max(diff(cu_seqlens))
```

### 13.3 同时传入不兼容的 `attention_mask`

传统 `[B, S]` padding mask 与 `[1, T]` flattened 输入的语义不同。某些框架在收到 `attention_mask` 后会走普通 unpadding 路径、忽略 `position_ids` 边界或退回非预期后端。

### 13.4 边界参数被 Trainer 删除

Trainer 可能根据 model forward signature 删除“不使用”的 dataset 列。如果 `seq_lengths`、`seq_idx` 或其他边界字段没有出现在签名或保留列表中，collator/模型可能静默退化。

### 13.5 后端静默回退

模型可能因 dtype、GPU 架构、head dimension、窗口设置或未安装扩展而从边界感知 kernel 回退到参考实现。对 conv、GDN、SSM 尤其要确认回退实现是否仍尊重 `seq_idx`。

### 13.6 位置编号不匹配

输出形状正常但 eval 大幅下降时，应检查位置起点、RoPE offset、多模态位置张量形状和每段重置规则。

### 13.7 Loss 分母错误

当 gradient accumulation 改变后 loss 数值按累积步数漂移，或不同 pack 组成导致训练结果明显变化，应检查是否对每个 micro-batch 先独立求 mean。

### 13.8 对齐 padding 破坏后端

TP、SP、CP 可能要求 token 数或单条序列长度对齐。样本间插入的 padding 与普通尾部 padding 语义不同；Transformer Engine 某些 THD 配置在 `pad_between_seqs=True` 时可能禁用 FlashAttention 后端。

---

## 十四、工程落地路线

### 14.1 第一步：建立 baseline

记录：

- 样本长度分布；
- 平均 padding 比例；
- 每秒有效 token/帧；
- 峰值显存；
- 训练和 eval 指标；
- attention、MLP、conv 和数据加载耗时。

没有 baseline，就无法判断 packing 是否真正加速，或只是改变了 batch/token 口径。

### 14.2 第二步：先做分桶与动态批

对长度方差大的任务，先按 token、帧或音频秒数控制 batch 容量。这一步不改变模型语义，通常最容易稳定落地。

### 14.3 第三步：只改标准 Transformer

先在纯 self-attention + MLP 模型上接入 collator 与 varlen backend，并完成 packed/unpacked 等价性测试。

### 14.4 第四步：处理 labels、loss 与位置编码

显式处理跨边界标签、有效 token 分母和位置起点。不要把这些逻辑隐含在数据格式假设中。

### 14.5 第五步：扩展到 conv、线性注意力和 SSM

逐个确认所有时序混合算子是否消费 `seq_idx` 或边界元数据。无法确认时，优先采用分层打包，而不是假设 position IDs 足够。

### 14.6 第六步：扩展到分布式并行

分别验证 TP、SP、CP、PP 下的元数据传递、长度对齐、局部索引和 loss 归一化。单卡正确不代表多卡切分后仍正确。

### 14.7 第七步：质量与性能联合验收

建议设置三组对照：

1. padded baseline；
2. 正确边界隔离的 packed 实现；
3. 故意不隔离边界的 naive 拼接，仅作为污染下界实验。

最终以 eval 质量不下降、有效吞吐提高、显存降低和结果顺序不变性共同作为验收标准。

---

## 十五、检查清单

### 数据与元数据

- [ ] `cu_seqlens` 为 `torch.int32`
- [ ] `cu_seqlens` 长度为 `B + 1`
- [ ] 首元素为 0，末元素为总 token 数
- [ ] `max_seqlen` 显式传入且值正确
- [ ] `position_ids` 按模型预训练约定生成
- [ ] 样本边界处的 labels 已屏蔽或正确插入 EOS
- [ ] 含 conv、线性注意力或 SSM 时提供 `seq_idx` 或等价边界

### 模型与内核

- [ ] attention backend 真正支持 varlen/THD
- [ ] 没有传入与 flattened 输入冲突的传统 padding mask
- [ ] 所有时间混合算子都尊重边界
- [ ] 后端回退路径同样尊重边界
- [ ] GQA/MQA 的接口和 head 数满足约束
- [ ] `torch.compile` 下没有因运行时边界推断产生 graph break

### 训练与并行

- [ ] loss 按有效目标总数归一化
- [ ] 已明确选择 token 等权或样本等权
- [ ] gradient accumulation 前后目标函数一致
- [ ] DP rank 之间按实际计算量平衡
- [ ] TP/SP/CP 的 token 和序列长度满足对齐要求
- [ ] PP 各 stage 获得一致的边界元数据

### 验证

- [ ] packed 与 padded 的有效位置输出一致
- [ ] 更换相邻样本后当前样本输出不变
- [ ] packed 与 padded 的梯度一致
- [ ] eval 指标不下降
- [ ] 吞吐以有效 token/帧/音频秒计量
- [ ] 峰值显存和端到端 wall-clock 均有记录

---

## 十六、适用边界与决策建议

### 优先采用

- 序列长度方差大；
- padding 比例高；
- 模型主要由标准 Transformer 层构成；
- attention 和激活显存是主要瓶颈；
- 已有 FlashAttention、Transformer Engine 或原生 varlen backend；
- 训练管线可以传递显式边界元数据。

### 谨慎采用

- 模型含大量卷积、线性注意力或 SSM；
- 需要复杂 streaming/chunk attention；
- 多模态 position IDs 结构复杂；
- 分布式并行对序列长度有严格对齐要求；
- 自定义 Trainer 可能丢弃 metadata；
- 现有测试无法验证跨样本独立性。

### 收益可能有限

- 数据长度非常均匀；
- attention 只占总运行时间很小一部分；
- batch 本身很小；
- 数据加载、通信或 CPU 预处理才是主要瓶颈；
- 为满足内核或 CP 对齐又重新加入大量 padding。

---

## 十七、总结

Packed-Input Batching 的本质可以概括为：

> 将 batch 中的有效 token 连续存放，用明确的序列边界驱动边界感知内核，从而消除 padding 计算，同时保持每条样本的独立语义。

对纯 Transformer，成熟方案已经包括 Hugging Face padding-free collator、FlashAttention varlen、Transformer Engine THD、FlexAttention 和 PyTorch 原生 varlen attention。正确实现的关键不是“拼接成功”，而是：

1. `cu_seqlens`、`max_seqlen` 和 `seq_idx` 等元数据准确；
2. 每一个跨时间混合信息的算子都尊重样本边界；
3. labels、position IDs 和 loss 分母与打包后的布局一致；
4. 分布式切分和后端回退不会丢失边界；
5. 通过 packed/unpacked 等价性与独立 eval 指标验证质量。

对于语音、Conformer 和多模态模型，卷积前端与流式掩码使问题明显复杂。更稳妥的路径是先做分桶和动态批处理，再从纯 Transformer/LLM 层开始分层打包，最后逐步引入边界感知卷积或专用 kernel。

训练侧 sequence packing、推理侧 continuous batching 和 PagedAttention 共享“只处理有效工作量”的目标，但分别解决数据布局、在线调度和 KV cache 管理问题。准确区分这些层次，是设计和排查高效变长系统的前提。

---

## 参考资料

### Packed training 与变长注意力

1. Hugging Face, [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2)
2. Kundu et al., [Efficient Data Packing for Large Language Models](https://arxiv.org/abs/2407.09105)
3. Hugging Face Transformers, [Padding-free training](https://huggingface.co/docs/transformers/en/padding_free)
4. Dao-AILab, [FlashAttention](https://github.com/Dao-AILab/flash-attention)
5. NVIDIA, [Transformer Engine Attention](https://nvidia.github.io/TransformerEngine/examples/attention/attention.html)
6. NVIDIA, [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
7. PyTorch Tutorials, [Using Variable Length Attention](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)

### 正确性与已知问题

8. Hugging Face TRL, [PR #3673](https://github.com/huggingface/trl/pull/3673)
9. Hugging Face TRL, [PR #3699](https://github.com/huggingface/trl/pull/3699)
10. Hugging Face Transformers, [Issue #39814](https://github.com/huggingface/transformers/issues/39814)
11. FlashAttention, [Issue #2381](https://github.com/Dao-AILab/flash-attention/issues/2381)
12. sentence-transformers, [v5.6.1 release notes](https://newreleases.io/project/github/huggingface/sentence-transformers/release/v5.6.1)
13. NVIDIA BioNeMo, [Scale Biology Transformer Models with PyTorch and NVIDIA BioNeMo Recipes](https://developer.nvidia.com/blog/scale-biology-transformer-models-with-pytorch-and-nvidia-bionemo-recipes/)

### 卷积与语音

14. Friso et al., [RubiConv: Efficient Boundary-Respecting Convolutions](https://arxiv.org/abs/2605.08451)
15. Gulati et al., [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
16. Lhotse, [A Speech Data Representation Library for the Modern Deep Learning Ecosystem](https://arxiv.org/abs/2110.12561)
17. NVIDIA NeMo, [ASR Datasets](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html)
18. SpeechBrain, [DynamicBatchSampler](https://speechbrain.readthedocs.io/en/v0.5.13/API/speechbrain.dataio.sampler.html)
19. TouchNet, [Efficient Sequence Packing with FlexAttention](https://github.com/xingchensong/TouchNet)

### 大规模并行与推理

20. NVIDIA, [Dynamic Context Parallelism with Megatron Core](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core)
21. NVIDIA NeMo, [PR #9540: Context Parallel support for packed THD datasets](https://github.com/NVIDIA/NeMo/pull/9540)
22. vLLM, [PagedAttention](https://docs.vllm.ai/)
23. TensorRT-LLM, [Batching](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)

> 注：性能数字、接口名称和后端限制会随框架版本变化。落地时应以目标版本的官方文档、源码和最小复现实验为准。
