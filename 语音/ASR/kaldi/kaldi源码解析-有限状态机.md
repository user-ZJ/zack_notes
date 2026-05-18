# kaldi源码解析-有限状态机

## 概述

Kaldi 的 `fstext` 模块是语音识别系统中处理加权有限状态转换器（Weighted Finite State Transducers, WFST）的核心组件。该模块基于 OpenFST 库构建，提供了一系列针对语音识别场景优化的 FST 操作和数据结构。

## 模块架构

### 核心组件层次

```
fstext/
├── fstext-lib.h          # 主头文件，包含所有核心组件引用
├── kaldi-fst-io.h/.cc    # FST 输入输出操作
├── context-fst.h/.cc     # 上下文相关 FST（HCLG 中的 C）
├── deterministic-fst.h   # 确定性按需 FST 接口
├── fstext-utils.h        # FST 工具函数
├── lattice-weight.h      # Lattice 权重类型
├── determinize-star.h    # DeterminizeStar 算法
├── determinize-lattice.h # Lattice 确定化
├── lattice-utils.h       # Lattice 操作工具
└── table-matcher.h       # 表匹配器
```

## 核心组件详解

### FST I/O 操作

#### 主要函数

|            函数名            |                    功能描述                    |
| ---------------------------- | ---------------------------------------------- |
| `ReadFstKaldi()`             | 读取二进制 FST 文件，支持管道等 Kaldi I/O 机制 |
| `ReadFstKaldiGeneric()`      | 通用 FST 读取函数，支持 ConstFst 和 VectorFst  |
| `WriteFstKaldi()`            | 写入 FST 文件                                  |
| `CastOrConvertToVectorFst()` | 将 FST 强制转换或转换为 VectorFst              |
| `ReadAndPrepareLmFst()`      | 读取语言模型 FST 并转换为 acceptor             |

#### VectorFstTplHolder 类

用于支持 Kaldi Table 机制的 FST Holder 类，实现了 `Holder` 接口：

```cpp
template<class Arc>
class VectorFstTplHolder {
public:
    typedef VectorFst<Arc> T;
    bool Write(std::ostream &os, bool binary, const T &t);
    bool Read(std::istream &is);
    T &Value();
    void Swap(VectorFstTplHolder<Arc> *other);
};
```

### 上下文相关 FST（Context FST）

#### 概念介绍

Context FST（通常称为 C.fst）是 HCLG 解码图中的核心组件，负责将**音素上下文窗口**（如三音素 `a,b,c`）转换为**单个音素**（如 `a`）。

#### 核心类：InverseContextFst

```cpp
class InverseContextFst : public DeterministicOnDemandFst<StdArc> {
public:
    InverseContextFst(Label subsequential_symbol,        // '$' 符号
                      const std::vector<int32>& phones,   // 音素列表
                      const std::vector<int32>& disambig_syms,  // 消歧符号
                      int32 context_width,               // 上下文宽度（如三音素为3）
                      int32 central_position);           // 中心位置（如三音素为1）
};
```

#### 关键参数说明

|          参数          |           说明           |                示例值                 |
| ---------------------- | ------------------------ | ------------------------------------- |
| `context_width`        | 上下文窗口大小           | 3（三音素）、2（双音素）、1（单音素） |
| `central_position`     | 中心音素位置（零基索引） | 1（三音素）、0（单音素）              |
| `subsequential_symbol` | 后续符号 '$'             | 通常为未使用的整数ID                  |
| `disambig_syms`        | 消歧符号列表             | #0, #1, #2 等的整数ID                 |

#### 核心函数

```cpp
// 组合上下文 FST 与输入 FST
void ComposeContext(const std::vector<int32> &disambig_syms,
                    int32 context_width,
                    int32 central_position,
                    VectorFst<StdArc> *ifst,      // 输入 FST（如 LG.fst）
                    VectorFst<StdArc> *ofst,      // 输出 FST（如 CLG.fst）
                    std::vector<std::vector<int32> > *ilabels_out);

// 添加后续循环
void AddSubsequentialLoop(StdArc::Label subseq_symbol,
                          MutableFst<StdArc> *fst);
```

### 确定性按需 FST 接口

#### DeterministicOnDemandFst 基类

这是一个轻量级的 FST 接口，假设每个状态对于给定输入符号只有一条弧：

```cpp
template<class Arc>
class DeterministicOnDemandFst {
public:
    virtual StateId Start() = 0;
    virtual Weight Final(StateId s) = 0;
    virtual bool GetArc(StateId s, Label ilabel, Arc *oarc) = 0;
};
```

#### 派生类体系

|               类名                |            功能            |
| --------------------------------- | -------------------------- |
| `BackoffDeterministicOnDemandFst` | 带回溯的语言模型包装器     |
| `ScaleDeterministicOnDemandFst`   | 权重缩放（如 LM 权重调整） |
| `UnweightedNgramFst`              | 无权重 n-gram 历史 FST     |
| `ComposeDeterministicOnDemandFst` | 两个确定性 FST 的组合      |
| `CacheDeterministicOnDemandFst`   | 带缓存的确定性 FST         |


### Lattice 权重类型

#### LatticeWeightTpl

用于存储两个浮点值的权重类型，通常分别表示**声学得分**和**语言模型得分**：

```cpp
template<class FloatType>
class LatticeWeightTpl {
public:
    LatticeWeightTpl(T a, T b);  // value1_, value2_
    T Value1() const;            // 通常为声学得分
    T Value2() const;            // 通常为语言模型得分
};
```

**半环运算**：
- **Plus**：取更优路径（较小的 `value1_ + value2_`）
- **Times**：`(a1+a2, b1+b2)`
- **Compare**：先比较 `value1_+value2_`，再比较 `value1_-value2_`

#### CompactLatticeWeightTpl

紧凑 Lattice 权重，包含权重值和符号序列：

```cpp
template<class WeightType, class IntType>
class CompactLatticeWeightTpl {
public:
    const W &Weight() const;                    // 权重值
    const std::vector<IntType> &String() const; // 符号序列
};
```


### FST 工具函数

#### 符号操作

|              函数               |        功能        |
| ------------------------------- | ------------------ |
| `HighestNumberedOutputSymbol()` | 获取最大输出符号ID |
| `HighestNumberedInputSymbol()`  | 获取最大输入符号ID |
| `GetInputSymbols()`             | 获取输入符号列表   |
| `GetOutputSymbols()`            | 获取输出符号列表   |
| `ClearSymbols()`                | 清除符号           |

#### FST 变换

|                函数                |               功能               |
| ---------------------------------- | -------------------------------- |
| `DeterminizeStarInLog()`           | 在对数半环中执行 DeterminizeStar |
| `PushInLog()`                      | 在对数半环中执行 Push 操作       |
| `MinimizeEncoded()`                | 编码后最小化                     |
| `SafeDeterminizeWrapper()`         | 安全的确定化包装器               |
| `SafeDeterminizeMinimizeWrapper()` | 安全确定化并最小化               |

#### 特殊操作

|          函数          |             功能             |
| ---------------------- | ---------------------------- |
| `MakeLinearAcceptor()` | 创建线性 acceptor            |
| `MakeLoopFst()`        | 创建循环 FST                 |
| `PhiCompose()`         | Phi 组合（用于回退语言模型） |
| `RhoCompose()`         | Rho 组合                     |
| `PropagateFinal()`     | 传播终态概率                 |
| `IsStochasticFst()`    | 检查 FST 是否随机            |


## HCLG 解码图构建流程

```
L.fst (Lexicon)          G.fst (Language Model)
       |                      |
       +----------+-----------+
                  |
                  v
               LG.fst (组合)
                  |
                  v
            AddSubsequentialLoop
                  |
                  v
            ComposeContext (C.fst)
                  |
                  v
               CLG.fst
                  |
                  v
            Compose with H.fst
                  |
                  v
               HCLG.fst (最终解码图)
```

---

## 核心算法

### DeterminizeStar 算法

针对带消歧符号的 FST 进行确定化，处理 epsilon 转换和消歧符号的特殊情况：

```cpp
void DeterminizeStarInLog(VectorFst<StdArc> *fst, 
                          float delta = kDelta, 
                          bool *debug_ptr = NULL,
                          int max_states = -1);
```

### Lattice 确定化

将 Lattice 转换为确定化形式，用于高效的最佳路径搜索：

```cpp
void DeterminizeLattice(const Lattice &ifst, Lattice *ofst, ...);
```

## 使用示例

### 读取和写入 FST

```cpp
// 读取 FST
fst::Fst<fst::StdArc> *fst = fst::ReadFstKaldiGeneric("model.fst");

// 转换为 VectorFst
fst::VectorFst<fst::StdArc> *vfst = fst::CastOrConvertToVectorFst(fst);

// 写入 FST
fst::WriteFstKaldi(*vfst, "output.fst");
```

### 构建 Context FST

```cpp
std::vector<int32> phones = {1, 2, 3, 4, 5};      // 音素列表
std::vector<int32> disambig_syms = {100, 101};     // 消歧符号
int32 context_width = 3;                           // 三音素
int32 central_position = 1;                        // 中心位置

fst::InverseContextFst ctx_fst(
    subsequential_symbol,
    phones,
    disambig_syms,
    context_width,
    central_position
);
```

### FST 组合

```cpp
fst::VectorFst<fst::StdArc> lg_fst, clg_fst;
std::vector<std::vector<int32> > ilabels;

// 添加后续循环
fst::AddSubsequentialLoop(subsequential_symbol, &lg_fst);

// 组合上下文
fst::ComposeContext(disambig_syms, context_width, 
                    central_position, &lg_fst, &clg_fst, &ilabels);
```



## 设计要点

### 按需构建（On-Demand）

`DeterministicOnDemandFst` 接口允许按需构建状态和弧，避免预先展开整个 FST，节省内存并提高效率。

### 对数半环优化

在对数半环中执行确定化等操作，避免数值下溢问题：

```cpp
template<ReweightType rtype>
void PushInLog(VectorFst<StdArc> *fst, uint32 ptype, float delta = kDelta) {
    VectorFst<LogArc> *fst_log = new VectorFst<LogArc>;
    Cast(*fst, fst_log);
    // 在对数半环中操作...
}
```

### 消歧符号处理

消歧符号（#0, #1, #2...）用于解决 FST 组合后的非确定化问题，在 Context FST 中通过自环传递。

### 随机化保证

`IsStochasticFst()` 函数验证 FST 的随机性质，确保每个状态的输出弧权重和为 1。


## 性能优化策略

|    优化手段    |                         说明                         |
| -------------- | ---------------------------------------------------- |
| **ConstFst**   | 使用只读 FST 提高解码性能                            |
| **缓存机制**   | `CacheDeterministicOnDemandFst` 缓存常用弧           |
| **编码最小化** | `MinimizeEncoded()` 在保持符号位置的同时最小化状态数 |
| **对数半环**   | 避免数值精度问题                                     |



## 相关工具

### fstbin 目录下的命令行工具

|           工具            |      功能      |
| ------------------------- | -------------- |
| `fstcopy`                 | 复制 FST       |
| `fstcomposecontext`       | 组合上下文 FST |
| `fstdeterminizelog`       | 对数确定化     |
| `fstminimizeencoded`      | 编码最小化     |
| `fstaddselfloops`         | 添加自环       |
| `fstaddsubsequentialloop` | 添加后续循环   |
| `fstmakecontextfst`       | 构建上下文 FST |
| `fsttablecompose`         | 表组合         |


## 总结

Kaldi 的 `fstext` 模块提供了完整的 WFST 操作工具链，特别针对语音识别场景进行了优化：

1. **Context FST**：高效处理音素上下文建模
2. **确定性按需接口**：支持大规模语言模型的高效查询
3. **Lattice 权重系统**：灵活处理多维度得分
4. **丰富的工具函数**：支持各种 FST 变换和优化

该模块是 Kaldi 语音识别系统的核心组件，为解码图构建和声学解码提供了坚实的基础。