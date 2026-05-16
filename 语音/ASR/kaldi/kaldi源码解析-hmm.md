# kaldi源码解析-HMM

## HMM 基本概念

HMM（Hidden Markov Model，隐马尔可夫模型）是一种用于建模时序数据的概率模型，特别适合语音识别中的声学建模。它由两个随机过程组成：

- 隐状态序列（hidden states）：这些状态不可直接观察，代表音素或音素状态。
- 观测序列（observations）：可见数据，例如 MFCC、Fbank 等声学特征。

HMM 的基本要素包括：

- 初始状态概率：模型从某个隐藏状态开始的概率分布。
- 状态转移概率：从一个隐藏状态转移到另一个隐藏状态的概率。
- 发射概率（观测概率）：在某个隐藏状态下生成某个观测值的概率。

HMM 的三大经典问题是：

1. 概率计算问题：给定模型和观测序列，计算观测序列的概率。
2. 解码问题：求最可能的隐藏状态序列（通常使用 Viterbi 算法）。
3. 学习问题：在观测数据下估计模型参数（如 Baum-Welch / EM 算法）。

在语音识别中，HMM 常用来表示音素或上下文相关音素的时间结构，每个音素由若干个 emitting state 组成，state 之间通过 self-loop 和前向转移连接。Kaldi 在此基础上引入 `pdf-id` 和 `transition-id` 等机制，将传统 HMM 与解码图、决策树共享模型结合起来。

## 概述

在 Kaldi 中，HMM（Hidden Markov Model，隐马尔可夫模型）是声学模型与解码图之间的核心结构。Kaldi 把 HMM 的三部分设计为：

- 拓扑结构（topology）
- 转移概率（transition probabilities）
- 发射概率（emission probabilities / pdf）

Kaldi 的 HMM 代码主要分布在 `src/hmm/` 目录，核心文件包括：

- `src/hmm/hmm-topology.h` / `src/hmm/hmm-topology.cc`
- `src/hmm/transition-model.h` / `src/hmm/transition-model.cc`
- `src/hmm/hmm-utils.h`

这些文件定义了 HMM 的拓扑、转移 ID、pdf-id、phone-id 等关系，并且连接到解码图中的 FST 输入标签。

---

## Kaldi 中的 HMM 术语

### Phone、pdf、transition-id

- `phone`：语音单元（可以是上下文相关的 triphone state）。在 Kaldi 中，`phone` 通常指经过决策树变换后的状态单元，例如 `Sil`, `ah_0` 等。
- `pdf-id`：概率密度函数索引。`pdf-id` 代表一个声学状态对应的高斯混合模型或神经网络输出分类。
- `transition-id`：转移 ID，是 HMM 过渡弧的唯一编号。Kaldi 将每个 pdf 的 self-loop 和 forward-transition 编号为不同的 transition-id。

### `TransitionModel`

`TransitionModel` 是 Kaldi HMM 的核心类，负责把 `phone`、`pdf-id`、`transition-id` 关系封装起来。它的作用包括：

- 存储各 phone 的 HMM 拓扑结构
- 提供 `transition-id -> pdf-id` 映射
- 提供 `transition-id -> phone / state` 映射
- 维护转移概率 (`transition probs`) 和转移类别

常见函数：

- `TransitionIdToPdf(transition_id)`
- `TransitionIdToPhone(transition_id)`
- `TransitionIdToHmmState(transition_id)`
- `NumTransitionIdsForPdf(pdf_id)`
- `TransitionId(transition_model, pdf_id, transition_index)`

这些映射在解码图构建、对齐计算和声学评分时都必须使用。

### `HmmTopology`

`HmmTopology` 表示 phone 的内部 HMM 拓扑结构，例如每个 phone 有多少个 emitting-state、是否有 self-loop、是否允许 epsilon 转移、状态间的连接方式等。

Kaldi 采用拓扑文件（`.topo`）或直接用 `HmmTopology` 结构初始化。该拓扑通常描述：

- phone 的 state 数
- 每个 state 的 pdf-class
- 每个 state 的 outgoing transitions 和概率

在 `src/hmm/hmm-topology.h` 中，`TopologyEntry` 和 `TopologyState` 描述了这些信息。

---

## Kaldi HMM 源码关键结构

### `src/hmm/hmm-topology.h`

这个文件定义了 HMM 拓扑的内存结构。主要结构包括：

- `TopologyState`：一个 HMM 状态的信息
  - `pdf_class`：该状态属于哪个 pdf 类
  - `transitions`：一组 outgoing transitions
- `TopologyEntry`：一个 phone 的拓扑定义
  - `phone`：phone ID
  - `states`：该 phone 的状态列表

### `src/hmm/hmm-topology.cc`

实现细节包括：

- 从 `.topo` 文件读取拓扑
- 将拓扑写回 `.topo` 文件
- `HmmTopology::Check()`：检查拓扑是否合法
- 拓扑在 training graph 编译前的初始化过程

这一层主要负责“状态级别的 HMM 拓扑定义”，它不直接涉及 pdf-id 和 transition-id 的编号。

### `src/hmm/transition-model.h`

`TransitionModel` 的声明文件，核心字段包括：

- `transition_ids_` / `pdf_to_transition_ids_`
- `phone2pdf_`：phone 到 pdf-id 的映射表
- `transition_probs_`：每个 transition 的概率表
- `phone_map_`：用于把 phone-id 映射到树或决策树类别的结构

`TransitionModel` 提供的典型接口：

- `NumTransitionIds()`
- `NumPdfs()`
- `TransitionIdToPdf(tid)`
- `TransitionIdToPhone(tid)`
- `TransitionIdToHmmState(tid)`
- `TransitionIdToTransitionIndex(tid)`
- `PdfToTransitionIds(pdf_id, &vec)`

这些接口在 `src/fstext` 的图编译、解码器和对齐过程里频繁使用。

### `src/hmm/transition-model.cc`

这个文件实现了 `TransitionModel` 的构造与映射算法，关键逻辑包括：

- `TransitionModel::Init(...)`：根据 topology 和 phone2pdf 生成 transition-id 编号
- `TransitionModel::ComputeTransitionId(pdf, transition_index)`：计算 transition-id
- `TransitionModel::TransitionIdToPdf()` / `TransitionIdToPhone()` / `TransitionIdToHmmState()` 等映射实现
- `TransitionModel::Write()` / `Read()`：序列化模型到文件

这部分代码会把“逻辑上的 HMM 转移关系”变为“可以用于 FST 的整数 ID”。

---

## Kaldi HMM 的编号规则与意义

### transition-id 的编号规则

Kaldi 对于一个 HMM 状态的转移弧，会产生一组 transition-id。通常一个 emitting state 的 outgoing transitions 有两类：

- self-loop 转移
- forward 转移（到下一个 HMM state）

因此对于一个 `pdf-id`，会有多个 transition-id。它们的编号方式通常固定，且与 `transition_index` 相关。

这使得：

- `transition-id` 反映 HMM 中任意一条弧
- `pdf-id` 反映发射概率类别

在解码 FST 中，输入标签通常使用 `pdf-id`，而输出标签则使用 `word-id`；转移弧的 `transition-id` 则用于对齐结果和训练过程中的状态序列。

### pdf-id 与 phone 的关系

`TransitionModel` 维护 `phone2pdf_`：它把每个 context-dependent phone 映射到一系列 `pdf-id`。

- 对于每个 phone，有一个按顺序排列的 HMM 状态序列
- 每个状态对应一个 `pdf-class`
- 这些 `pdf-class` 在训练时被树绑定成 pdf-id

这意味着：

- phone 的每个 emitting state 都对应一个 pdf-id
- 不同 phone 的同类状态可能通过决策树共享同一个 pdf-id

---

## Kaldi HMM 在训练和解码中的位置

### 训练阶段

训练流程中，HMM 负责把音素序列与声学特征对齐：

1. `compile-train-graph` 根据 `tree`、`0.mdl`、`L.fst` 生成训练图
2. 训练图中的输入符号是 `pdf-id`
3. HMM 拓扑决定了每个 phone 的状态数及转移模式
4. 对齐算法（如 Viterbi 或 forward-backward）使用 `TransitionModel` 的 transition-id 映射
5. `ali-to-post` / `align-equal-compiled` 等程序会生成基于 transition-id 的对齐结果

Kaldi 训练时最关键的是：

- `HMM topology` 决定了每个音素的状态结构
- `TransitionModel` 决定了 transition-id/pd-fid 的编号
- GMM/NN 模型输出的是 pdf-id 概率

### 解码阶段

在解码图 `HCLG.fst` 中，HMM 作用于声学模型与语言模型之间：

- `H`：表示 HMM 发射模型，通过输入符号 `pdf-id` 与 `TransitionModel` 绑定
- `C`：表示上下文依赖和决策树状态映射
- `L`：表示发音词典
- `G`：表示语言模型

解码器读取的发射分数与 `pdf-id` 对应，解码器内部会把 `transition-id` 映射回 `phone` 和 `hmm state`，用于输出对齐、后验统计、fMLLR、lattice 等。

### 对齐与后验统计

Kaldi 在对齐和训练统计中，常见的计算方式是：

- `transition-id` 作为对齐结果的最小单位
- `pdf-id` 作为声学打分的索引

这也是 Kaldi 区别于传统 HMM 的一个点：Kaldi 强调“transition-id 作为状态转移的唯一标识”，而不是仅仅使用 phone-state。

---

## HMM 相关文件与流程对应

### `src/hmm/hmm-topology.{h,cc}`

作用：

- 解析 `.topo` 文件
- 定义每个 phone 的 HMM 状态和 transition 结构
- 检查拓扑合法性

适用场景：

- 初始化 monophone HMM
- 初始化上下文相关 HMM 的通用拓扑
- `compile-train-graph` 之前

### `src/hmm/transition-model.{h,cc}`

作用：

- 根据 topology 和 `phone2pdf` 生成 transition-id 编号
- 提供转换接口：`transition-id <-> pdf-id`, `transition-id <-> phone`, `transition-id <-> hmm state`
- 读写 `final.mdl` 等模型文件

适用场景：

- 训练模型 `gmm-est` / `nnet3-train`
- 解码和对齐时的 transition-id 解析

### 其他相关文件

- `src/hmm/hmm-utils.h`：提供 HMM 相关工具函数，例如索引映射、输出读取、转移概率调整等
- `src/fstext/fstext-utils.h`：与 HMM FST 生成结合，产生 `H` 图层
- `src/decoder/`：实际解码器使用 `TransitionModel` 对输出标签做映射

---

## Kaldi HMM 的实现要点

### 拓扑灵活性

Kaldi 的 HMM 拓扑不是固定 3 state mono-phone，而是支持任意拓扑：

- 每个 phone 可以有 3 个 emitting state，也可以有 5 个或 1 个
- 每个 state 可以有不同的 transition 结构
- 通过 `.topo` 文件可以灵活定义

这在 `src/hmm/hmm-topology.cc` 中体现为拓扑读取和状态转移检查。

### transition-id 机制

这是 Kaldi HMM 的核心创新点之一。`transition-id` 将 HMM 转移弧与“pdf/phone”绑定起来：

- 一个 `pdf-id` 可能对应多个 transition-id
- transition-id 还可以用来表示“self-loop”与“forward transition”
- 它支持对齐结果保存为一个整数序列，方便训练和后处理

### phone 与 pdf 的分离

Kaldi 把“phone-state”与“pdf-id”分离开来：

- `phone` 表示 HMM 结构和 phonetic context
- `pdf-id` 表示实际的发射分布（可共享、可绑树）

这使得 Kaldi 能够高效实现决策树音素模型和模型参数共享。

### IO 与序列化

`TransitionModel` 与 `HmmTopology` 都支持读写接口：

- `TransitionModel::Write()` / `Read()`
- `HmmTopology::Write()` / `Read()`

模型文件一般存储于 `final.mdl`、`tree`、`topo` 等文件中。程序启动时读取这些文件，构建与解码器共享的 HMM 结构。

---

## 关键概念与 Kaldi HMM 关系图

### HMM 的三层关系

1. `phone` → HMM 拓扑（由 `HmmTopology` 定义）
2. HMM 状态 → `pdf-class`
3. `pdf-class` → `pdf-id` → 发射分布

然后 `TransitionModel` 负责把这些关系映射到 FST 可用的 `transition-id`。

### 训练/解码流程中的 HMM

- `prepare-training-graph`：把 `L.fst` 和 `H` 结合成 `HCLG.fst`
- `H` 的输入符号是 `pdf-id`
- `H` 的输出符号通常是 `epsilon`
- 解码时，`acoustic scorer` 计算 `pdf-id` 概率并交给解码器
- 对齐/训练时，`transition-id` 决定具体的 HMM 状态转移路径

---

## 读源码时重点关注的函数

推荐阅读以下函数与代码片段：

- `TransitionModel::Init(...)`：初始化 transition 模型的核心入口
- `TransitionModel::TransitionIdToPdf(...)` / `TransitionIdToPhone(...)` / `TransitionIdToHmmState(...)`
- `HmmTopology::Read()` / `HmmTopology::Check()`
- `HmmTopology::Write()`
- `TransitionModel::Write()` / `TransitionModel::Read()`
- `GetPdfInfo` / `Phone2Pdf` 等函数

这些代码路径直接体现 Kaldi 如何将 HMM 概念映射到实际模型结构。

---

## 结论

Kaldi 的 HMM 实现比传统“固定 3-state HMM”更灵活，其核心思想是：

- 用 `HmmTopology` 表示任意 HMM 状态拓扑
- 用 `TransitionModel` 将 HMM 转移弧映射到唯一的 `transition-id`
- 用 `pdf-id` 表示实际发射概率类别
- 通过 `transition-id` 让对齐、训练、解码过程之间建立统一接口

如果要深入阅读 Kaldi 源码，建议先从 `src/hmm/hmm-topology.cc` 和 `src/hmm/transition-model.cc` 入手，再结合 `src/tree/`、`src/fstext/` 和 `src/decoder/` 的调用链理解整体流程。