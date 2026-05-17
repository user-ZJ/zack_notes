# 解码图构建（HCLG）详解

## 引言

解码图（HCLG）是 Kaldi 语音识别系统的核心组件，它将声学模型、上下文信息、词典和语言模型组合成一个统一的有限状态转换器。解码图的质量直接影响识别准确率和解码效率。本文将深入探讨 HCLG 的组成部分和构建流程。

## HCLG 的组成

### HCLG 的含义

HCLG 是由四个 FST 组合而成的解码图：

| 组件 | 含义 | 输入符号 | 输出符号 |
|------|------|----------|----------|
| **H** | HMM 拓扑 | transition-id | 上下文相关音素（senone） |
| **C** | 上下文扩展 | 上下文无关音素（phone） | 上下文相关音素（senone） |
| **L** | 词典 | 音素（phone） | 词（word） |
| **G** | 语言模型 | 词（word） | 词（word） |

### 组合关系

```
输入: transition-id → H → senone → C → phone → L → word → G → word → 输出: word
```

**数学表示**：
$$\text{HCLG} = \text{H} \circ \text{C} \circ \text{L} \circ \text{G}$$

其中 $\circ$ 表示 FST 的组合操作。

### 各组件的作用

**H（HMM Topology FST）**：
- 描述音素的状态转移结构
- 输入：transition-id
- 输出：senone（上下文相关音素状态）

**C（Context Dependency FST）**：
- 将上下文无关音素扩展为上下文相关音素
- 处理三音素上下文（左、中、右）
- 输入：上下文无关音素
- 输出：上下文相关音素（senone）

**L（Lexicon FST）**：
- 描述词到音素序列的映射
- 包含发音变体
- 处理词边界标记
- 输入：音素序列
- 输出：词序列

**G（Language Model FST）**：
- 描述词序列的概率分布
- 通常是 N-gram 语言模型
- 输入：词
- 输出：词

## H 组件：HMM 拓扑 FST

### H 的构建

```cpp
void GetHTransducer(const TransitionModel &trans_model,
                    const ContextDependency &ctx_dep,
                    fst::VectorFst<fst::StdArc> *H) {
    // 获取所有可能的 transition-ids
    std::vector<int32> trans_ids = trans_model.GetTransitionIds();
    
    // 创建状态
    int32 start_state = H->AddState();
    int32 final_state = H->AddState();
    
    H->SetStart(start_state);
    H->SetFinal(final_state, 0.0);
    
    // 添加转移
    for (int32 trans_id : trans_ids) {
        // 获取 transition-id 对应的信息
        int32 trans_state = trans_model.TransitionIdToTransitionState(trans_id);
        int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
        BaseFloat log_prob = trans_model.LogProb(trans_id);
        
        // 获取 senone（上下文相关音素状态）
        int32 senone = pdf_id;
        
        // 添加弧：transition-id → senone
        fst::StdArc arc(
            trans_id,           // 输入：transition-id
            senone,             // 输出：senone
            log_prob,           // 权重：转移概率
            final_state         // 下一状态（简化版）
        );
        
        H->AddArc(start_state, arc);
    }
}
```

### H 的简化表示

```
状态 0（起始）
  ├─(trans_id=1, senone=100, weight=-0.5)→ 状态 1（终止）
  ├─(trans_id=2, senone=100, weight=-0.3)→ 状态 1（终止）
  ├─(trans_id=3, senone=101, weight=-0.4)→ 状态 1（终止）
  └─(trans_id=4, senone=101, weight=-0.6)→ 状态 1（终止）
状态 1（终止） weight=0.0
```

## C 组件：上下文扩展 FST

### C 的作用

将上下文无关音素扩展为上下文相关音素（senone）：

```
输入：/t/（上下文无关）
输出：/p/_t_/a/（上下文相关三音素）
```

### C 的构建

```cpp
void GetCTransducer(const ContextDependency &ctx_dep,
                    fst::VectorFst<fst::StdArc> *C) {
    // 获取所有可能的上下文组合
    std::vector<ContextType> contexts = GetAllContexts(ctx_dep);
    
    int32 start_state = C->AddState();
    int32 final_state = C->AddState();
    
    C->SetStart(start_state);
    C->SetFinal(final_state, 0.0);
    
    for (const ContextType &ctx : contexts) {
        // 获取中心音素
        int32 central_phone = ctx.phones[ctx_dep.CentralPosition()];
        
        // 获取 senone（通过决策树）
        int32 senone = ctx_dep.ContextToPdf(ctx);
        
        // 添加弧：phone → senone
        fst::StdArc arc(
            central_phone,      // 输入：上下文无关音素
            senone,             // 输出：上下文相关音素
            0.0,                // 权重：无额外代价
            final_state         // 下一状态
        );
        
        C->AddArc(start_state, arc);
    }
}
```

### 上下文窗口示例

```
上下文窗口宽度 = 3，中心位置 = 1

上下文组合：
- [sil, aa, sil] → senone 1
- [sil, aa, ae] → senone 2
- [p, aa, sil] → senone 3
- [p, aa, ae] → senone 4
...
```

## L 组件：词典 FST

### L 的结构

词典 FST 描述词到音素序列的映射：

```
状态 0
  ├─(sil, <eps>, 0.0)→ 状态 1
  ├─(aa, <eps>, 0.0)→ 状态 2
  └─(p, <eps>, 0.0)→ 状态 3
状态 1
  ├─(</s>, </s>, 0.0)→ 状态 4（终止）
状态 2
  ├─(ae, <word1>, 0.0)→ 状态 4
  └─(sil, <word2>, 0.0)→ 状态 4
状态 3
  ├─(aa, <word3>, 0.0)→ 状态 4
  └─(ae, <word4>, 0.0)→ 状态 4
状态 4（终止） weight=0.0
```

### L 的构建

```cpp
void BuildLexiconFst(const Lexicon &lexicon,
                     fst::VectorFst<fst::StdArc> *L) {
    // 添加特殊符号
    int32 sil_id = lexicon.GetPhoneId("sil");
    int32 sp_id = lexicon.GetPhoneId("sp");
    int32 eps_id = 0;
    int32 bos_id = lexicon.GetWordId("<s>");
    int32 eos_id = lexicon.GetWordId("</s>");
    
    // 创建状态
    int32 start_state = L->AddState();
    int32 sil_state = L->AddState();
    int32 final_state = L->AddState();
    
    L->SetStart(start_state);
    L->SetFinal(final_state, 0.0);
    
    // 添加静音自环
    fst::StdArc sil_arc(sil_id, eps_id, 0.0, sil_state);
    L->AddArc(start_state, sil_arc);
    L->AddArc(sil_state, sil_arc);  // 自环
    
    // 添加词的发音
    for (const auto &entry : lexicon.entries()) {
        std::string word = entry.first;
        int32 word_id = lexicon.GetWordId(word);
        const std::vector<std::vector<int32>> &pronunciations = entry.second;
        
        for (const auto &pron : pronunciations) {
            int32 curr_state = start_state;
            
            for (size_t i = 0; i < pron.size(); i++) {
                int32 phone = pron[i];
                
                // 最后一个音素输出词，其他输出 epsilon
                int32 olabel = (i == pron.size() - 1) ? word_id : eps_id;
                
                // 创建下一个状态
                int32 next_state = L->AddState();
                
                fst::StdArc arc(phone, olabel, 0.0, next_state);
                L->AddArc(curr_state, arc);
                
                curr_state = next_state;
            }
            
            // 添加到终止状态的弧
            fst::StdArc final_arc(eps_id, eps_id, 0.0, final_state);
            L->AddArc(curr_state, final_arc);
        }
    }
    
    // 添加词边界标记
    AddWordBoundaries(L, lexicon);
}
```

### 发音变体处理

```cpp
// 同一个词可能有多种发音
lexicon.add("tomato", {"t", "aa", "m", "ey", "t", "ow"});
lexicon.add("tomato", {"t", "ah", "m", "ey", "t", "ow"});
```

## G 组件：语言模型 FST

### N-gram 语言模型

语言模型描述词序列的概率：

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^n P(w_i | w_{i-k+1}, ..., w_{i-1})$$

### G 的构建

```cpp
void BuildLmFst(const LanguageModel &lm,
                fst::VectorFst<fst::StdArc> *G) {
    // 获取所有词
    std::vector<std::string> words = lm.GetWords();
    
    // 创建状态（每个状态对应一个历史）
    std::map<std::vector<std::string>, int32> state_map;
    int32 start_state = G->AddState();
    state_map[{}] = start_state;
    
    G->SetStart(start_state);
    
    // 遍历所有 N-gram
    for (const auto &ngram : lm.GetNgrams()) {
        std::vector<std::string> history = ngram.history;
        std::string word = ngram.word;
        BaseFloat log_prob = ngram.log_prob;
        BaseFloat backoff = ngram.backoff;
        
        // 获取历史状态
        int32 history_state = state_map[history];
        
        // 获取下一个历史状态
        std::vector<std::string> next_history = history;
        next_history.push_back(word);
        if (next_history.size() > lm.GetOrder()) {
            next_history.erase(next_history.begin());
        }
        
        // 创建新状态（如果不存在）
        if (state_map.find(next_history) == state_map.end()) {
            state_map[next_history] = G->AddState();
        }
        int32 next_state = state_map[next_history];
        
        // 添加弧
        int32 word_id = lm.GetWordId(word);
        fst::StdArc arc(word_id, word_id, log_prob, next_state);
        G->AddArc(history_state, arc);
        
        // 添加回退弧
        if (backoff != 0.0 && history.size() > 0) {
            std::vector<std::string> backoff_history = history;
            backoff_history.erase(backoff_history.begin());
            int32 backoff_state = state_map[backoff_history];
            
            fst::StdArc backoff_arc(0, 0, backoff, backoff_state);
            G->AddArc(history_state, backoff_arc);
        }
    }
    
    // 设置终止状态
    for (const auto &pair : state_map) {
        G->SetFinal(pair.second, 0.0);
    }
}
```

### 语言模型缩放

```cpp
// 将语言模型权重缩放到合适范围
void ScaleLm(fst::VectorFst<fst::StdArc> *G, BaseFloat scale) {
    for (int32 state = 0; state < G->NumStates(); state++) {
        fst::MutableArcIterator<fst::VectorFst<fst::StdArc>> aiter(G, state);
        
        for (; !aiter.Done(); aiter.Next()) {
            fst::StdArc arc = aiter.Value();
            arc.weight *= scale;
            aiter.SetValue(arc);
        }
    }
}
```

## HCLG 组合流程

### 组合顺序

```
1. 构建 H、C、L、G 四个 FST
2. 组合 H ○ C → HC
3. 组合 HC ○ L → HCL
4. 组合 HCL ○ G → HCLG
5. 优化：确定化、最小化、添加自环
```

### 组合实现

```cpp
void ComposeHclg(const TransitionModel &trans_model,
                 const ContextDependency &ctx_dep,
                 const fst::Fst<fst::StdArc> &L,
                 const fst::Fst<fst::StdArc> &G,
                 fst::VectorFst<fst::StdArc> *HCLG) {
    // 1. 构建 H
    fst::VectorFst<fst::StdArc> H;
    GetHTransducer(trans_model, ctx_dep, &H);
    
    // 2. 构建 C
    fst::VectorFst<fst::StdArc> C;
    GetCTransducer(ctx_dep, &C);
    
    // 3. 组合 H ○ C
    fst::VectorFst<fst::StdArc> HC;
    fst::Compose(H, C, &HC);
    
    // 4. 组合 HC ○ L
    fst::VectorFst<fst::StdArc> HCL;
    fst::Compose(HC, L, &HCL);
    
    // 5. 组合 HCL ○ G
    fst::Compose(HCL, G, HCLG);
    
    // 6. 优化
    OptimizeFst(HCLG);
}

void OptimizeFst(fst::VectorFst<fst::StdArc> *fst) {
    // 移除 epsilon 转换
    fst::RmEpsilon(fst);
    
    // 确定化
    fst::DeterminizeOptions det_opts;
    det_opts.max_states = 1000000;
    fst::Determinize(*fst, fst, det_opts);
    
    // 最小化
    fst::Minimize(fst);
    
    // 移除无用状态
    fst::RemoveDeadStates(fst);
}
```

## 自环添加

### 为什么需要自环

在解码过程中，每个时间帧需要处理一个特征向量，对应一个 HMM 状态转移。自环允许在同一状态停留多帧：

```
状态 A →(self-loop)→ 状态 A（停留）
状态 A →(transition)→ 状态 B（转移）
```

### 自环添加实现

```cpp
void AddSelfLoops(const TransitionModel &trans_model,
                  const std::vector<int32> &disambig_syms,
                  BaseFloat self_loop_scale,
                  bool reorder,
                  fst::VectorFst<fst::StdArc> *fst) {
    // 收集所有需要添加自环的状态
    std::set<int32> states_with_self_loops;
    for (int32 trans_id = 1; trans_id <= trans_model.NumTransitionIds(); trans_id++) {
        int32 trans_state = trans_model.TransitionIdToTransitionState(trans_id);
        int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
        
        // 检查是否需要为该状态添加自环
        if (NeedsSelfLoop(trans_model, trans_state)) {
            states_with_self_loops.insert(pdf_id);
        }
    }
    
    // 为每个状态添加自环
    for (int32 state = 0; state < fst->NumStates(); state++) {
        fst::ArcIterator<fst::StdArc> aiter(*fst, state);
        
        for (; !aiter.Done(); aiter.Next()) {
            const fst::StdArc &arc = aiter.Value();
            
            // 如果弧的输出是需要自环的状态
            if (states_with_self_loops.count(arc.olabel)) {
                // 创建自环弧
                fst::StdArc self_loop_arc(
                    0,                          // 输入：epsilon
                    0,                          // 输出：epsilon
                    self_loop_scale * GetSelfLoopWeight(trans_model, arc.olabel),
                    state                       // 下一状态：自身
                );
                
                fst->AddArc(state, self_loop_arc);
            }
        }
    }
    
    // 可选：重排序自环
    if (reorder) {
        ReorderSelfLoops(fst);
    }
}
```

## 实际构建流程

### 准备工作

```bash
# 1. 准备词典
utils/prepare_lang.sh \
    data/local/dict \
    "<UNK>" \
    data/local/lang \
    data/lang

# 2. 编译语言模型
utils/format_lm.sh \
    data/lang \
    data/local/lm/lm.arpa.gz \
    data/local/dict/lexicon.txt \
    data/lang_test

# 3. 准备上下文相关模型
steps/train_deltas.sh \
    --nj 4 \
    data/train \
    data/lang \
    exp/mono \
    exp/tri1
```

### 构建 HCLG

```bash
# 使用 mkgraph.sh 脚本构建
utils/mkgraph.sh \
    --self-loop-scale 0.1 \
    data/lang_test \
    exp/tri1 \
    exp/tri1/graph

# 手动构建流程
# 1. 构建 L.fst
fstcompile --isymbols=data/lang_test/phones.txt \
           --osymbols=data/lang_test/words.txt \
           data/lang_test/L.txt \
           data/lang_test/L.fst

# 2. 构建 G.fst
utils/make_kn_lm.sh data/local/lm/lm.arpa.gz data/lang_test

# 3. 编译 HCLG
compile-graphs \
    --transition-model=exp/tri1/final.mdl \
    --context-dep=exp/tri1/tree \
    --lex-fst=data/lang_test/L.fst \
    --lm-fst=data/lang_test/G.fst \
    exp/tri1/graph/HCLG.fst

# 4. 添加自环
add-self-loops \
    --self-loop-scale=0.1 \
    exp/tri1/graph/HCLG.fst \
    exp/tri1/graph/HCLG.fst
```

## HCLG 优化策略

### 内存优化

```cpp
// 使用 ConstFst 减少内存占用
fst::VectorFst<fst::StdArc> *vector_fst = ReadFstKaldi("HCLG.fst");
fst::ConstFst<fst::StdArc> *const_fst = new fst::ConstFst<fst::StdArc>(*vector_fst);
```

### 弧排序优化

```cpp
// 按输入标签排序，加速解码时的查找
void SortArcsByInputLabel(fst::VectorFst<fst::StdArc> *fst) {
    for (int32 state = 0; state < fst->NumStates(); state++) {
        std::vector<fst::StdArc> arcs;
        
        // 收集所有弧
        fst::ArcIterator<fst::StdArc> aiter(*fst, state);
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
}
```

### 剪枝优化

```cpp
// 移除低概率路径
void PruneLowProbabilityArcs(fst::VectorFst<fst::StdArc> *fst,
                              BaseFloat threshold) {
    for (int32 state = 0; state < fst->NumStates(); state++) {
        fst::MutableArcIterator<fst::VectorFst<fst::StdArc>> aiter(fst, state);
        
        for (; !aiter.Done();) {
            const fst::StdArc &arc = aiter.Value();
            
            if (arc.weight > threshold) {
                aiter.Remove();  // 移除高权重（低概率）的弧
            } else {
                aiter.Next();
            }
        }
    }
}
```

## HCLG 统计信息

### 查看统计信息

```bash
fstinfo exp/tri1/graph/HCLG.fst
```

### 统计示例

```
HCLG FST 统计信息：
- 状态数：1,234,567
- 弧数：4,567,890
- 输入符号数：12,345（transition-ids）
- 输出符号数：10,000（词）
- 平均每状态弧数：3.7
- 最大状态深度：42
- 内存占用：512 MB
```

## 总结

HCLG 解码图是 Kaldi 语音识别的核心：

1. **模块化设计**：将声学模型、词典、语言模型分离
2. **统一框架**：使用 FST 组合所有组件
3. **高效解码**：通过优化和剪枝提高解码效率
4. **灵活扩展**：支持多种语言模型和词典格式

理解 HCLG 的构建过程是掌握 Kaldi 解码流程的关键。下一篇我们将深入探讨解码器的实现原理。

---

**系列文章目录**：
1. Kaldi 整体架构概览
2. 特征提取模块详解
3. HMM 拓扑与转移模型
4. 高斯混合模型（GMM）实现
5. 决策树与上下文相关建模
6. 有限状态转换器（FST）集成
7. 解码图构建（HCLG）详解（本文）
8. 解码器实现原理
9. 神经网络声学模型
10. 训练流程与工具链
11. Chain 模型与 Lattice-Free MMI
12. 在线解码与流式识别