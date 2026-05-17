# 有限状态转换器（FST）集成

## 引言

有限状态转换器（Finite State Transducer, FST）是 Kaldi 语音识别系统的核心组件之一。它提供了一种统一的数学框架，用于表示和操作解码图。Kaldi 利用 FST 将 HMM 拓扑、上下文扩展、词典和语言模型组合成一个高效的解码图（HCLG）。本文将深入探讨 Kaldi 中 FST 的集成方式和核心实现。

## FST 基础概念

### 什么是有限状态转换器

FST 是一种有限状态机，它将输入符号序列转换为输出符号序列：

```
状态 →(输入:输出/权重)→ 状态
```

**核心组成**：
- **状态集合**：有限个状态
- **初始状态**：转换的起点
- **终止状态**：转换的终点
- **弧（Arc）**：状态之间的转换，包含输入符号、输出符号和权重
- **权重半环**：用于计算路径权重（通常是热带半环或对数半环）

### FST 的数学定义

$$F = (Q, \Sigma, \Delta, q_0, F, E)$$

其中：
- $Q$：状态集合
- $\Sigma$：输入字母表
- $\Delta$：输出字母表
- $q_0 \in Q$：初始状态
- $F \subseteq Q$：终止状态集合
- $E \subseteq Q \times (\Sigma \cup \{\epsilon\}) \times (\Delta \cup \{\epsilon\}) \times \mathbb{R} \times Q$：弧的集合

### 常用操作

**1. 组合（Composition）**
$$F_1 \circ F_2$$
将两个 FST 串联，第一个的输出作为第二个的输入。

**2. 确定化（Determinization）**
将非确定型 FST 转换为确定型 FST。

**3. 最小化（Minimization）**
找到等价的最小状态数 FST。

**4. 投影（Projection）**
提取输入或输出投影。

**5. 连接（Concatenation）**
$$F_1 \cdot F_2$$
将两个 FST 顺序连接。

**6. 闭包（Closure）**
$$F^*$$
允许 FST 重复任意次数（包括零次）。

## OpenFst 库介绍

### Kaldi 与 OpenFst

Kaldi 使用 OpenFst 库作为 FST 的底层实现：

```cpp
// Kaldi 中常用的 FST 类型
typedef fst::VectorFst<fst::StdArc> StdFst;
typedef fst::VectorFst<fst::LogArc> LogFst;

// Arc 类型
struct StdArc {
    int32 ilabel;      // 输入标签
    int32 olabel;      // 输出标签
    float weight;       // 权重（热带半环）
    int32 nextstate;   // 下一状态
};

struct LogArc {
    int32 ilabel;      // 输入标签
    int32 olabel;      // 输出标签
    double weight;      // 权重（对数半环，负值表示概率）
    int32 nextstate;   // 下一状态
};
```

### 半环选择

**热带半环（Tropical Semiring）**：
- 加法：$\oplus = \min$
- 乘法：$\otimes = +$
- 单位元：$\mathbf{0} = \infty$，$\mathbf{1} = 0$
- 用于最短路径问题

**对数半环（Log Semiring）**：
- 加法：$\oplus = \log(\exp(a) + \exp(b))$
- 乘法：$\otimes = a + b$
- 用于概率计算（避免数值下溢）

## Kaldi 中的 FST 封装

### KaldiFst 类

```cpp
class KaldiFst {
public:
    // 创建 FST
    static KaldiFst* Read(std::istream &is, bool binary);
    static KaldiFst* ReadText(std::istream &is);
    
    // 获取底层 FST
    virtual const fst::Fst<fst::StdArc>* GetFst() const = 0;
    virtual fst::MutableFst<fst::StdArc>* GetMutableFst() = 0;
    
    // 写入 FST
    virtual void Write(std::ostream &os, bool binary) const = 0;
};
```

### FstLib 类

```cpp
class FstLib : public KaldiFst {
public:
    FstLib(fst::VectorFst<fst::StdArc> *fst) : fst_(fst) {}
    
    const fst::Fst<fst::StdArc>* GetFst() const override { return fst_.get(); }
    fst::MutableFst<fst::StdArc>* GetMutableFst() override { return fst_.get(); }
    
    void Write(std::ostream &os, bool binary) const override {
        fst_->Write(os, binary);
    }
    
private:
    std::unique_ptr<fst::VectorFst<fst::StdArc>> fst_;
};
```

### FST 辅助函数

```cpp
// 读取 FST
fst::VectorFst<fst::StdArc>* ReadFstKaldi(std::string rxfilename);

// 写入 FST
void WriteFstKaldi(const fst::VectorFst<fst::StdArc> &fst, std::string wxfilename);

// 组合两个 FST
void ComposeFsts(const fst::Fst<fst::StdArc> &fst1,
                 const fst::Fst<fst::StdArc> &fst2,
                 fst::VectorFst<fst::StdArc> *out);

// 确定化
void DeterminizeFst(const fst::Fst<fst::StdArc> &ifst,
                    fst::VectorFst<fst::StdArc> *ofst);

// 最小化
void MinimizeFst(fst::VectorFst<fst::StdArc> *fst);

// 添加自环
void AddSelfLoops(const TransitionModel &trans_model,
                  fst::VectorFst<fst::StdArc> *fst);
```

## FST 操作详解

### 组合操作

```cpp
void ComposeFsts(const fst::Fst<fst::StdArc> &fst1,
                 const fst::Fst<fst::StdArc> &fst2,
                 fst::VectorFst<fst::StdArc> *out) {
    // 使用 OpenFst 的 Composition 操作
    fst::Compose(fst1, fst2, out);
    
    // Kaldi 特有的优化：移除epsilon传递
    RemoveEpsilon(out);
    
    // 确定化和最小化
    DeterminizeFst(*out, out);
    MinimizeFst(out);
}
```

### 确定化算法

```cpp
void DeterminizeFst(const fst::Fst<fst::StdArc> &ifst,
                    fst::VectorFst<fst::StdArc> *ofst) {
    // 使用 OpenFst 的 Determinize 操作
    fst::DeterminizeOptions opts;
    opts.max_states = 1000000;  // 最大状态数限制
    opts.max_arcs = 10000000;   // 最大弧数限制
    
    fst::Determinize(ifst, ofst, opts);
}
```

### 最小化算法

```cpp
void MinimizeFst(fst::VectorFst<fst::StdArc> *fst) {
    // 使用 OpenFst 的 Minimize 操作
    fst::Minimize(fst);
}
```

## FST 输入输出格式

### Kaldi 特有的 FST 格式

**文本格式**：
```
0 1 1 1 0.5
1 2 2 2 0.3
2 0.1
```

解释：
- `状态 下一状态 输入标签 输出标签 权重`
- 最后一行是终止状态和终止权重

**二进制格式**：
使用 OpenFst 的二进制格式，包含：
- FST 类型标识
- 状态数
- 弧的数量
- 每个状态的弧列表
- 终止状态信息

### 符号表管理

```cpp
class SymbolTable {
public:
    // 添加符号
    void AddSymbol(std::string symbol, int32 id);
    
    // 获取符号对应的 ID
    int32 Find(std::string symbol) const;
    
    // 获取 ID 对应的符号
    std::string Find(int32 id) const;
    
    // 读取/写入
    void Read(std::istream &is);
    void Write(std::ostream &os) const;
};
```

**符号表示例**（phones.txt）：
```
<eps> 0
sil 1
sp 2
aa 3
ae 4
...
```

## FST 在解码中的应用

### HCLG 解码图

Kaldi 的解码图由四个 FST 组合而成：

```
H ○ C ○ L ○ G
```

**H（HMM 拓扑）**：描述音素的状态转移
**C（上下文扩展）**：将单音素扩展为上下文相关音素
**L（词典）**：音素到词的映射
**G（语言模型）**：词序列的概率模型

### HCLG 构建流程

```cpp
void BuildHclg(const TransitionModel &trans_model,
               const ContextDependency &ctx_dep,
               const fst::Fst<fst::StdArc> &L,
               const fst::Fst<fst::StdArc> &G,
               fst::VectorFst<fst::StdArc> *HCLG) {
    // 1. 构建 H（HMM 拓扑 FST）
    fst::VectorFst<fst::StdArc> H;
    GetHTransducer(trans_model, ctx_dep, &H);
    
    // 2. 构建 C（上下文扩展 FST）
    fst::VectorFst<fst::StdArc> C;
    GetCTransducer(ctx_dep, &C);
    
    // 3. 组合 H ○ C
    fst::VectorFst<fst::StdArc> HC;
    ComposeFsts(H, C, &HC);
    
    // 4. 组合 HC ○ L
    fst::VectorFst<fst::StdArc> HCL;
    ComposeFsts(HC, L, &HCL);
    
    // 5. 组合 HCL ○ G
    ComposeFsts(HCL, G, HCLG);
    
    // 6. 优化
    RemoveEpsilon(HCLG);
    DeterminizeFst(*HCLG, HCLG);
    MinimizeFst(HCLG);
}
```

### 自环添加

```cpp
void AddSelfLoops(const TransitionModel &trans_model,
                  const std::vector<int32> &disambig_syms,
                  BaseFloat self_loop_scale,
                  bool reorder,
                  fst::VectorFst<fst::StdArc> *fst) {
    // 为每个状态添加自环
    for (int32 state = 0; state < fst->NumStates(); state++) {
        // 获取该状态对应的 transition-ids
        std::vector<int32> trans_ids = GetTransitionIdsForState(trans_model, state);
        
        for (int32 trans_id : trans_ids) {
            // 获取转移概率
            BaseFloat log_prob = trans_model.LogProb(trans_id);
            
            // 创建自环弧
            fst::StdArc arc(
                trans_id,                  // 输入标签（transition-id）
                0,                          // 输出标签（无输出）
                self_loop_scale * log_prob, // 缩放后的权重
                state                       // 下一状态（自身）
            );
            
            fst->AddArc(state, arc);
        }
    }
    
    // 可选：重排序自环到目标状态
    if (reorder) {
        ReorderSelfLoops(fst);
    }
}
```

## 核心文件解析

### fstext-utils.cc

**主要功能**：FST 操作的工具函数

```cpp
// 移除 epsilon 转换
void RemoveEpsilon(fst::MutableFst<fst::StdArc> *fst) {
    fst::RmEpsilon(fst);
}

// 反转 FST
void ReverseFst(const fst::Fst<fst::StdArc> &ifst,
                fst::VectorFst<fst::StdArc> *ofst) {
    fst::Reverse(ifst, ofst);
}

// 交集操作
void IntersectFsts(const fst::Fst<fst::StdArc> &fst1,
                   const fst::Fst<fst::StdArc> &fst2,
                   fst::VectorFst<fst::StdArc> *out) {
    fst::Intersect(fst1, fst2, out);
}
```

### kaldi-fst-io.cc

**主要功能**：FST 的输入输出

```cpp
// 读取 FST（自动检测格式）
fst::VectorFst<fst::StdArc>* ReadFstKaldi(std::string rxfilename) {
    std::unique_ptr<std::istream> is = OpenInputFile(rxfilename);
    
    // 检测是否为二进制格式
    bool binary = DetectBinary(*is);
    
    if (binary) {
        fst::VectorFst<fst::StdArc> *fst = new fst::VectorFst<fst::StdArc>();
        fst->Read(*is);
        return fst;
    } else {
        // 文本格式
        return fst::ReadFstKaldiText(*is);
    }
}

// 写入 FST
void WriteFstKaldi(const fst::VectorFst<fst::StdArc> &fst,
                   std::string wxfilename) {
    std::unique_ptr<std::ostream> os = OpenOutputFile(wxfilename);
    fst.Write(*os, true);  // 默认二进制格式
}
```

### compile-graphs.cc

**主要功能**：编译解码图

```cpp
int main(int argc, char *argv[]) {
    // 解析命令行参数
    ParseOptions po("Compile HCLG graph");
    
    std::string trans_model_rxfilename;
    std::string ctx_dep_rxfilename;
    std::string lex_fst_rxfilename;
    std::string lm_fst_rxfilename;
    std::string hclg_wxfilename;
    
    po.Register("transition-model", &trans_model_rxfilename, "Transition model");
    po.Register("context-dep", &ctx_dep_rxfilename, "Context dependency");
    po.Register("lex-fst", &lex_fst_rxfilename, "Lexicon FST");
    po.Register("lm-fst", &lm_fst_rxfilename, "Language model FST");
    po.Register("hclg", &hclg_wxfilename, "Output HCLG FST");
    
    po.Read(argc, argv);
    
    // 读取输入
    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);
    
    ContextDependency ctx_dep;
    ReadKaldiObject(ctx_dep_rxfilename, &ctx_dep);
    
    fst::VectorFst<fst::StdArc> *L = ReadFstKaldi(lex_fst_rxfilename);
    fst::VectorFst<fst::StdArc> *G = ReadFstKaldi(lm_fst_rxfilename);
    
    // 构建 HCLG
    fst::VectorFst<fst::StdArc> HCLG;
    BuildHclg(trans_model, ctx_dep, *L, *G, &HCLG);
    
    // 添加自环
    AddSelfLoops(trans_model, disambig_syms, 0.1, true, &HCLG);
    
    // 写入输出
    WriteFstKaldi(HCLG, hclg_wxfilename);
    
    return 0;
}
```

## 实际应用示例

### 构建解码图

```bash
# 准备词典 FST
utils/prepare_lang.sh \
    data/local/dict \
    "<UNK>" \
    data/local/lang \
    data/lang

# 编译语言模型
utils/format_lm.sh \
    data/lang \
    data/local/lm/lm.arpa.gz \
    data/local/dict/lexicon.txt \
    data/lang_test

# 构建 HCLG 图
utils/mkgraph.sh \
    --self-loop-scale 0.1 \
    data/lang_test \
    exp/tri3b \
    exp/tri3b/graph
```

### FST 操作示例

```bash
# 查看 FST 统计信息
fstinfo exp/tri3b/graph/HCLG.fst

# 转换为文本格式
fstprint exp/tri3b/graph/HCLG.fst HCLG.txt

# 绘制 FST 图
fstdraw exp/tri3b/graph/HCLG.fst | dot -Tpng -o HCLG.png

# 组合两个 FST
fstcompose L.fst G.fst LG.fst

# 确定化
fstdeterminize LG.fst LG_det.fst

# 最小化
fstminimize LG_det.fst LG_min.fst
```

## 性能优化策略

### FST 压缩

```cpp
// 使用 ConstFst 减少内存占用
fst::VectorFst<fst::StdArc> *vector_fst = ReadFstKaldi("HCLG.fst");
fst::ConstFst<fst::StdArc> *const_fst = new fst::ConstFst<fst::StdArc>(*vector_fst);
```

### 弧排序

```cpp
// 按输入标签排序弧，加速查找
for (int32 state = 0; state < fst->NumStates(); state++) {
    fst::ArcIterator<fst::VectorFst<fst::StdArc>> aiter(*fst, state);
    std::vector<fst::StdArc> arcs;
    
    for (; !aiter.Done(); aiter.Next()) {
        arcs.push_back(aiter.Value());
    }
    
    // 按输入标签排序
    std::sort(arcs.begin(), arcs.end(),
              [](const fst::StdArc &a, const fst::StdArc &b) {
                  return a.ilabel < b.ilabel;
              });
    
    // 替换原弧
    fst->DeleteArcs(state);
    for (const auto &arc : arcs) {
        fst->AddArc(state, arc);
    }
}
```

### 懒加载

```cpp
// 只在需要时加载 FST
class LazyFstLoader {
public:
    LazyFstLoader(std::string filename) : filename_(filename), fst_(nullptr) {}
    
    const fst::Fst<fst::StdArc>* GetFst() {
        if (!fst_) {
            fst_.reset(ReadFstKaldi(filename_));
        }
        return fst_.get();
    }
    
private:
    std::string filename_;
    std::unique_ptr<fst::VectorFst<fst::StdArc>> fst_;
};
```

## FST 与解码器的交互

### 解码器中的 FST 使用

```cpp
class FasterDecoder {
public:
    FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                  const FasterDecoderOptions &opts)
        : fst_(fst), opts_(opts) {}
    
    void Decode(const MatrixBase<BaseFloat> &feats) {
        // 初始化搜索状态
        std::vector<DecoderState> states;
        states.push_back(DecoderState(fst_.Start(), 0.0));
        
        // 逐帧处理
        for (int32 frame = 0; frame < feats.NumRows(); frame++) {
            ProcessFrame(feats.Row(frame), &states);
        }
        
        // 获取最佳路径
        std::vector<int32> path;
        GetBestPath(&path);
    }
    
private:
    const fst::Fst<fst::StdArc> &fst_;
    FasterDecoderOptions opts_;
};
```

### 弧遍历与得分计算

```cpp
void ProcessFrame(const VectorBase<BaseFloat> &feat,
                  std::vector<DecoderState> *states) {
    std::vector<DecoderState> new_states;
    
    for (const auto &state : *states) {
        // 遍历当前状态的所有弧
        fst::ArcIterator<fst::Fst<fst::StdArc>> aiter(fst_, state.fst_state);
        
        for (; !aiter.Done(); aiter.Next()) {
            const fst::StdArc &arc = aiter.Value();
            
            // 计算声学得分
            BaseFloat acoustic_score = ComputeAcousticScore(arc.ilabel, feat);
            
            // 计算总得分
            BaseFloat total_score = state.score + arc.weight + acoustic_score;
            
            // 添加到新状态
            new_states.push_back(DecoderState(arc.nextstate, total_score));
        }
    }
    
    // Beam Search 剪枝
    PruneStates(&new_states);
    
    *states = std::move(new_states);
}
```

## 总结

Kaldi 的 FST 集成是其核心优势之一：

1. **统一框架**：将 HMM、词典、语言模型统一为 FST 操作
2. **高效计算**：利用 OpenFst 的优化实现
3. **灵活组合**：支持多种 FST 操作和组合方式
4. **可扩展性**：易于添加新的模型和算法
5. **优化工具**：提供完整的 FST 优化和压缩工具

理解 FST 在 Kaldi 中的应用是掌握解码流程的关键，也是深入理解语音识别系统的重要一步。

---

**系列文章目录**：
1. Kaldi 整体架构概览
2. 特征提取模块详解
3. HMM 拓扑与转移模型
4. 高斯混合模型（GMM）实现
5. 决策树与上下文相关建模
6. 有限状态转换器（FST）集成（本文）
7. 解码图构建（HCLG）详解
8. 解码器实现原理
9. 神经网络声学模型
10. 训练流程与工具链
11. Chain 模型与 Lattice-Free MMI
12. 在线解码与流式识别