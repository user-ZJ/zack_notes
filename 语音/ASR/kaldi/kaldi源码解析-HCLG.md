# kaldi源码解析-HCLG

## 概述

HCLG 是 Kaldi 语音识别框架中用于解码的核心有限状态转换器（Finite State Transducer, FST）。它是由四个组件组合而成的：

```
HCLG = H ○ C ○ L ○ G
```

其中 `○` 表示 FST 的组合操作（composition）。

## 各组件详解

### 1. G - 语言模型（Grammar/Language Model）

**定义**：G 是一个接受器（acceptor），即输入和输出符号相同，用于编码语言模型或语法规则。

**特性**：
- 输入/输出符号：词（words）
- 通常由 ARPA 格式的语言模型转换而来
- 包含特殊符号 `<s>`（句首）和 `</s>`（句尾）

**创建流程**：
```bash
gunzip -c data_prep/lm.arpa.gz | \
  arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=data/words.txt - data/G.fst
```

**关键技术点**：
- 使用 `#0` 消歧符号替换语言模型中的 epsilon 转换，确保 G 可确定性化
- 需要移除非法的句子符号序列（如 `<s>` 后跟 `</s>`）

### 2. L - 词典（Lexicon）

**定义**：L 是一个转换器，将音素序列映射到词序列。

**特性**：
- 输入符号：音素（phones）
- 输出符号：词（words）

**创建流程**：

首先创建不含消歧符号的词典 FST（用于训练）：
```bash
scripts/make_lexicon_fst.pl data/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=data/phones.txt --osymbols=data/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/L.fst
```

然后创建含消歧符号的词典 FST（用于解码）：
```bash
scripts/make_lexicon_fst.pl data/lexicon_disambig.txt 0.5 SIL | \
   fstcompile --isymbols=data/phones_disambig.txt --osymbols=data/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstaddselfloops <(echo $phone_disambig_symbol) <(echo $word_disambig_symbol) | \
   fstarcsort --sort_type=olabel > data/L_disambig.fst
```

**词典结构**：
- 包含一个循环状态（loop state），该状态同时也是终止状态
- 起始状态有两条转换到循环状态：一条带静音，一条不带
- 每个词对应一条转换，词作为输出符号，第一个音素作为输入符号
- 支持可选静音模型（optional silence）

### 3. C - 上下文相关模型（Context Dependency）

**定义**：C 是一个转换器，处理音素的上下文相关性。

**特性**：
- 输入符号：上下文相关音素（如三音素 a/b/c）
- 输出符号：上下文无关音素

**上下文窗口**：
- 参数 N：上下文宽度（通常为 3，即三音素模型）
- 参数 P：中心位置（通常为 1）

**特殊符号**：
- `#-1`：用于替代起始位置的 epsilon，确保可确定性化
- `$`（subsequential symbol）：用于处理句尾的上下文刷新

**动态创建**：
```bash
fstcomposecontext --read-disambig-syms=$dir/disambig_phones.list \
                  --write-disambig-syms=$dir/disambig_ilabels.list \
                  $dir/ilabels < $dir/LG.fst >$dir/CLG.fst
```

### 4. H - HMM 定义（Hidden Markov Model）

**定义**：H 是一个转换器，包含 HMM 的状态转换结构。

**特性**：
- 输入符号：转换标识（transition-ids），编码 pdf-id 和其他信息
- 输出符号：上下文相关音素

**创建流程**：
```bash
make-h-transducer --disambig-syms-out=$dir/disambig_tstate.list \
   --transition-scale=1.0 $dir/ilabels.remapped \
   $tree $model > $dir/Ha.fst
```

**H 的结构**：
- 包含一个既是初始状态也是终止状态的状态
- 每个上下文相关音素对应一个 HMM 结构（不含自环）
- 消歧符号在初始状态上有自环

## 完整构建流程

### 步骤 1：准备符号表

创建 `words.txt` 和 `phones.txt`，为所有词和音素分配整数 ID。OpenFst 保留 ID 0 用于 epsilon。

### 步骤 2：创建 LG

组合词典和语言模型：

```bash
fsttablecompose data/L_disambig.fst data/G.fst | \
    fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstpushspecial | \
    fstarcsort --sort-type=ilabel > somedir/LG.fst
```

**处理流程**：
1. 使用 `fsttablecompose` 进行高效组合
2. 使用 `fstdeterminizestar` 进行确定性化（带 epsilon 移除）
3. 使用 `fstminimizeencoded` 进行最小化（不进行权重推送）
4. 使用 `fstpushspecial` 调整权重分布
5. 使用 `fstarcsort` 对转换进行排序，加速后续操作

### 步骤 3：创建 CLG

添加上下文相关性：

```bash
fstcomposecontext --read-disambig-syms=$dir/disambig_phones.list \
                  --write-disambig-syms=$dir/disambig_ilabels.list \
                  $dir/ilabels < $dir/LG.fst >$dir/CLG.fst
```

**可选优化步骤**：减少上下文相关输入符号数量

```bash
make-ilabel-transducer --write-disambig-syms=$dir/disambig_ilabels_remapped.list \
  $dir/ilabels $tree $model $dir/ilabels.remapped > $dir/ilabel_map.fst

fstcompose $dir/ilabel_map.fst $dir/CLG.fst | \
  fstdeterminizestar --use-log=true | \
  fstminimizeencoded > $dir/CLG2.fst
```

### 步骤 4：创建 HCLG

组合 HMM 结构：

```bash
fsttablecompose $dir/Ha.fst $dir/CLG2.fst | \
   fstdeterminizestar --use-log=true | \
   fstrmsymbols $dir/disambig_tstate.list | \
   fstrmepslocal | fstminimizeencoded > $dir/HCLGa.fst
```

### 步骤 5：添加自环

```bash
add-self-loops --self-loop-scale=0.1 \
  --reorder=true $model < $dir/HCLGa.fst > $dir/HCLG.fst
```

**注意**：添加自环是唯一不保持随机性的步骤，因为 `self-loop-scale` 不为 1。

## 核心技术原理

### 消歧符号（Disambiguation Symbols）

**作用**：确保 FST 组合后的结果可确定性化。

**类型**：
- `#0`：用于 G.fst 的回退转换
- `#1, #2, #3...`：用于词典中具有相同音素序列前缀的词
- `#-1`：用于 C.fst 起始位置替代 epsilon

**原理**：当一个音素序列是另一个音素序列的前缀或出现在多个词中时，需要添加消歧符号。

### 随机性保持（Stochasticity Preservation）

**定义**：一个随机的 FST 是指从每个状态出发的所有转换概率（加上终止概率）之和为 1。

**重要性**：保持随机性可以加速搜索过程。

**策略**：
- 使用 log semiring 进行确定性化
- 使用不推送权重的最小化算法
- 使用局部 epsilon 移除算法

**验证工具**：`fstisstochastic` 用于检查 FST 的随机性。

### 确定性化（Determinization）

**定义**：将非确定性 FST 转换为确定性 FST。

**算法**：`fstdeterminizestar` 同时进行确定性化和 epsilon 移除。

**关键选项**：`--use-log=true` 确保在 log semiring 中操作，保持随机性。

### 最小化（Minimization）

**定义**：通过合并等价状态来减小 FST 大小。

**算法**：`fstminimizeencoded` 在不推送权重的情况下进行最小化。

**优势**：保持随机性，避免权重推送可能导致的问题。

## 训练时间 vs 测试时间

### 训练时间

训练时间的图构建更简单，主要差异：

1. **不需要消歧符号**：训练时 HCLG 是功能性的且无环
2. **G 是线性接受器**：对应训练转录文本
3. **无转移概率**：转移概率在解码时添加

**命令示例**：
```bash
compile-train-graphs $dir/tree $dir/1.mdl data/L.fst ark:data/train.tra \
   ark:$dir/graphs.fsts
```

### 测试时间

测试时间需要完整的 HCLG，包含所有消歧符号和语言模型。

## 关键工具总结

|         工具         |            功能             |
| -------------------- | --------------------------- |
| `arpa2fst`           | 将 ARPA 语言模型转换为 FST  |
| `fsttablecompose`    | 高效的 FST 组合             |
| `fstdeterminizestar` | 确定性化（带 epsilon 移除） |
| `fstminimizeencoded` | 最小化（不推送权重）        |
| `fstpushspecial`     | 特殊权重推送                |
| `fstcomposecontext`  | 动态组合上下文 FST          |
| `make-h-transducer`  | 创建 HMM 转换器             |
| `add-self-loops`     | 添加自环                    |
| `fstisstochastic`    | 检查随机性                  |

## 性能优化建议

1. **使用动态上下文组合**：避免显式创建完整的 C.fst
2. **合理设置概率尺度**：`self-loop-scale=0.1` 是常用值
3. **启用 reorder 选项**：`--reorder=true` 可提高解码速度（不适用于 Kaldi 原生解码器）
4. **定期检查随机性**：使用 `fstisstochastic` 确保中间结果的随机性

