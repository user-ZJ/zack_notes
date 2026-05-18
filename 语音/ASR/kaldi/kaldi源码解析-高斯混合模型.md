# 高斯混合模型（GMM）实现

## 引言

高斯混合模型（Gaussian Mixture Model, GMM）是语音识别中经典的声学模型，用于建模特征向量的概率分布。Kaldi 提供了高效的 GMM 实现，包括对角协方差和全协方差两种形式。本文将深入探讨 Kaldi 中 GMM 的设计原理和核心代码实现。

## GMM 的数学基础

### 模型定义

高斯混合模型是多个高斯分布的加权组合：

$$p(x|\lambda) = \sum_{i=1}^M w_i \cdot \mathcal{N}(x|\mu_i, \Sigma_i)$$

其中：
- $M$ 是混合分量数
- $w_i$ 是第 $i$ 个分量的权重，满足 $\sum_{i=1}^M w_i = 1$
- $\mathcal{N}(x|\mu_i, \Sigma_i)$ 是第 $i$ 个高斯分量
- $\mu_i$ 是均值向量
- $\Sigma_i$ 是协方差矩阵

### 高斯分布的两种形式

**1. 对角协方差高斯**
$$\mathcal{N}(x|\mu, \Sigma) = \frac{1}{(2\pi)^{D/2} \cdot |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

其中 $\Sigma$ 是对角矩阵，仅保留对角线元素。

**2. 全协方差高斯**
协方差矩阵是完整的对称正定矩阵，可以建模特征之间的相关性。

### EM 算法训练

GMM 使用期望最大化（Expectation-Maximization, EM）算法进行训练：

**E 步**：计算每个样本属于每个分量的后验概率
$$\gamma_{ji} = \frac{w_i \cdot \mathcal{N}(x_j|\mu_i, \Sigma_i)}{\sum_{k=1}^M w_k \cdot \mathcal{N}(x_j|\mu_k, \Sigma_k)}$$

**M 步**：更新模型参数
- 权重：$w_i = \frac{\sum_{j=1}^N \gamma_{ji}}{N}$
- 均值：$\mu_i = \frac{\sum_{j=1}^N \gamma_{ji} x_j}{\sum_{j=1}^N \gamma_{ji}}$
- 协方差：$\Sigma_i = \frac{\sum_{j=1}^N \gamma_{ji} (x_j-\mu_i)(x_j-\mu_i)^T}{\sum_{j=1}^N \gamma_{ji}}$

## Kaldi 中的 GMM 类设计

### DiagGmm 类

**对角协方差 GMM**，是 Kaldi 中最常用的形式：

```cpp
class DiagGmm {
public:
    // 初始化
    void Init(int32 num_comp, int32 dim);
    
    // 读取/写入
    void Read(std::istream &is, bool binary);
    void Write(std::ostream &os, bool binary) const;
    
    // 计算对数似然
    BaseFloat LogLikelihood(const VectorBase<BaseFloat> &data) const;
    
    // 计算后验概率
    void Posterior(const VectorBase<BaseFloat> &data,
                   VectorBase<BaseFloat> *posterior) const;
    
    // 获取分量
    const Gaussian &GetComponent(int32 i) const { return gauss_[i]; }
    Gaussian &GetComponent(int32 i) { return gauss_[i]; }
    
    // 分量数量和维度
    int32 NumGauss() const { return gauss_.size(); }
    int32 Dim() const { return dim_; }
    
private:
    std::vector<Gaussian> gauss_;      // 高斯分量列表
    Vector<BaseFloat> weights_;        // 分量权重（对数形式）
    int32 dim_;                        // 特征维度
};
```

### Gaussian 类

**单个高斯分量**：

```cpp
struct Gaussian {
    Vector<BaseFloat> mean;            // 均值向量
    Vector<BaseFloat> var;             // 方差向量（对角协方差）
    Vector<BaseFloat> inv_var;         // 方差的倒数
    BaseFloat log_det;                 // 协方差行列式的对数
};
```

### FullGmm 类

**全协方差 GMM**：

```cpp
class FullGmm {
public:
    void Init(int32 num_comp, int32 dim);
    
    BaseFloat LogLikelihood(const VectorBase<BaseFloat> &data) const;
    
    void Posterior(const VectorBase<BaseFloat> &data,
                   VectorBase<BaseFloat> *posterior) const;
    
private:
    std::vector<FullGaussian> gauss_;  // 全协方差高斯分量
    Vector<BaseFloat> weights_;        // 分量权重
    int32 dim_;                        // 特征维度
};
```

## GMM 核心实现

### 对数似然计算

```cpp
BaseFloat DiagGmm::LogLikelihood(const VectorBase<BaseFloat> &data) const {
    BaseFloat log_sum = -1.0e+10;  // 初始化为负无穷
    
    for (int32 i = 0; i < gauss_.size(); i++) {
        // 计算单个高斯分量的对数似然
        BaseFloat log_prob = LogLikelihoodForComponent(data, i);
        
        // log-sum-exp 技巧，避免数值下溢
        BaseFloat weight = weights_(i);
        log_sum = LogAdd(log_sum, weight + log_prob);
    }
    
    return log_sum;
}

BaseFloat DiagGmm::LogLikelihoodForComponent(const VectorBase<BaseFloat> &data,
                                              int32 comp) const {
    const Gaussian &g = gauss_[comp];
    
    // (x - mu)^T * Sigma^{-1} * (x - mu)
    BaseFloat mahalanobis = 0.0;
    for (int32 d = 0; d < dim_; d++) {
        BaseFloat diff = data(d) - g.mean(d);
        mahalanobis += diff * diff * g.inv_var(d);
    }
    
    // log(N(x|mu, Sigma)) = -0.5 * (D*log(2pi) + log_det + mahalanobis)
    return -0.5 * (dim_ * M_LOG_2PI + g.log_det + mahalanobis);
}
```

### 后验概率计算

```cpp
void DiagGmm::Posterior(const VectorBase<BaseFloat> &data,
                        VectorBase<BaseFloat> *posterior) const {
    // 计算每个分量的未归一化对数概率
    Vector<BaseFloat> log_probs(gauss_.size());
    for (int32 i = 0; i < gauss_.size(); i++) {
        log_probs(i) = weights_(i) + LogLikelihoodForComponent(data, i);
    }
    
    // 归一化：log(posterior) = log_prob - log_sum_exp(log_probs)
    BaseFloat log_sum = LogSumExp(log_probs);
    for (int32 i = 0; i < gauss_.size(); i++) {
        (*posterior)(i) = exp(log_probs(i) - log_sum);
    }
}
```

### EM 算法实现

```cpp
void MleDiagGmmUpdate(const DiagGmm &gmm,
                       const AccumDiagGmm &acc,
                       BaseFloat min_variance,
                       DiagGmm *out_gmm) {
    int32 num_comp = gmm.NumGauss();
    int32 dim = gmm.Dim();
    
    // 更新权重
    Vector<BaseFloat> weights(num_comp);
    BaseFloat total_count = acc.total_count();
    for (int32 i = 0; i < num_comp; i++) {
        weights(i) = log(acc.counts()(i) / total_count);
    }
    
    // 更新每个分量
    for (int32 i = 0; i < num_comp; i++) {
        Gaussian &g = out_gmm->GetComponent(i);
        BaseFloat count = acc.counts()(i);
        
        // 更新均值
        g.mean.CopyFromVec(acc.means().Row(i));
        g.mean.Scale(1.0 / count);
        
        // 更新方差（带最小值约束）
        Vector<BaseFloat> var(dim);
        var.CopyFromVec(acc.vars().Row(i));
        var.Scale(1.0 / count);
        for (int32 d = 0; d < dim; d++) {
            var(d) = std::max(var(d), min_variance);
        }
        g.var = var;
        
        // 更新逆方差和对数行列式
        g.inv_var.Resize(dim);
        BaseFloat log_det = 0.0;
        for (int32 d = 0; d < dim; d++) {
            g.inv_var(d) = 1.0 / g.var(d);
            log_det += log(g.var(d));
        }
        g.log_det = log_det;
    }
    
    out_gmm->SetWeights(weights);
}
```

## 高斯选择策略

### 为什么需要高斯选择

当 GMM 包含大量分量时（如 512 个），对每个特征向量计算所有分量的似然会非常耗时。高斯选择通过只计算一部分最相关的分量来加速。

### Kaldi 中的高斯选择方法

**1. 基于树的高斯选择（Tree-based Gaussian Selection）**

```cpp
class GaussCluster {
public:
    void BuildTree(const DiagGmm &gmm, int32 num_leaves);
    
    void Select(const VectorBase<BaseFloat> &data,
                int32 num_select,
                std::vector<int32> *selected) const;
};
```

**2. 对角高斯选择（Diagonal Gaussian Selection）**

```cpp
void SelectGaussiansDiag(const DiagGmm &gmm,
                         const VectorBase<BaseFloat> &data,
                         int32 num_select,
                         std::vector<std::pair<BaseFloat, int32>> *gauss_indices);
```

**3. 预选择（Pre-selection）**

先使用简化的距离度量快速筛选候选分量：

```cpp
// 使用均值向量的欧氏距离进行预选择
BaseFloat DistanceToMean(const VectorBase<BaseFloat> &data,
                         const VectorBase<BaseFloat> &mean) {
    BaseFloat dist = 0.0;
    for (int32 d = 0; d < data.Dim(); d++) {
        BaseFloat diff = data(d) - mean(d);
        dist += diff * diff;
    }
    return dist;
}
```

## GMM 在声学模型中的应用

### 状态级 GMM

每个 HMM 状态对应一个 GMM：

```cpp
class AmDiagGmm {
public:
    void Read(std::istream &is, bool binary);
    
    // 计算某个状态的对数似然
    BaseFloat LogLikelihood(int32 pdf_id,
                            const VectorBase<BaseFloat> &feat) const;
    
    // 获取指定 PDF 的 GMM
    const DiagGmm &GetPdf(int32 pdf_id) const { return pdfs_[pdf_id]; }
    
private:
    std::vector<DiagGmm> pdfs_;      // PDF 列表
    TransitionModel trans_model_;     // 转移模型
};
```

### 计算流程

```cpp
// 计算特征序列的对数似然
BaseFloat ComputeLogLikelihood(const AmDiagGmm &am,
                               const TransitionModel &trans_model,
                               const MatrixBase<BaseFloat> &feats,
                               const std::vector<int32> &alignment) {
    BaseFloat log_prob = 0.0;
    
    for (int32 t = 0; t < feats.NumRows(); t++) {
        int32 trans_id = alignment[t];
        int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
        log_prob += am.LogLikelihood(pdf_id, feats.Row(t));
    }
    
    return log_prob;
}
```

## 模型初始化与训练

### 初始化方法

**1. 随机初始化**
```cpp
void DiagGmm::RandInit(int32 num_comp, int32 dim,
                       BaseFloat var_floor = 0.001) {
    gauss_.resize(num_comp);
    weights_.Resize(num_comp);
    
    // 随机初始化均值（范围 [-1, 1]）
    for (int32 i = 0; i < num_comp; i++) {
        gauss_[i].mean.Resize(dim);
        gauss_[i].mean.SetRandn();
        gauss_[i].mean.Scale(2.0);
        gauss_[i].mean.Add(1.0);  // [-1, 1] -> [0, 2]
        
        // 初始方差设为 1.0
        gauss_[i].var.Resize(dim);
        gauss_[i].var.Set(1.0);
        
        // 计算逆方差和对数行列式
        ComputeInvVarsAndLogDets(&gauss_[i]);
    }
    
    // 均匀初始化权重
    weights_.Set(log(1.0 / num_comp));
}
```

**2. K-means 初始化**

使用 K-means 算法先进行聚类，然后用聚类中心作为初始均值：

```cpp
void InitFromKmeans(const MatrixBase<BaseFloat> &data,
                    int32 num_comp,
                    DiagGmm *gmm) {
    // 1. 运行 K-means 聚类
    std::vector<int32> assignments(data.NumRows());
    Matrix<BaseFloat> centers(num_comp, data.NumCols());
    Kmeans(data, num_comp, &assignments, &centers);
    
    // 2. 使用聚类结果初始化 GMM
    gmm->Init(num_comp, data.NumCols());
    
    for (int32 i = 0; i < num_comp; i++) {
        Gaussian &g = gmm->GetComponent(i);
        g.mean.CopyFromVec(centers.Row(i));
        g.var.Set(1.0);
        ComputeInvVarsAndLogDets(&g);
    }
    
    // 3. 均匀权重
    Vector<BaseFloat> weights(num_comp);
    weights.Set(log(1.0 / num_comp));
    gmm->SetWeights(weights);
}
```

### 训练流程

```bash
# 1. 初始化 GMM
steps/train_mono.sh \
    --nj 4 \
    --totgauss 1000 \
    data/train \
    data/lang \
    exp/mono

# 2. 增加高斯分量
steps/train_deltas.sh \
    --nj 4 \
    --totgauss 2000 \
    data/train \
    data/lang \
    exp/mono \
    exp/tri1

# 3. LDA+MLLT 训练
steps/train_lda_mllt.sh \
    --nj 4 \
    --totgauss 4000 \
    data/train \
    data/lang \
    exp/tri1 \
    exp/tri2b

# 4. SAT 训练
steps/train_sat.sh \
    --nj 4 \
    --totgauss 8000 \
    data/train \
    data/lang \
    exp/tri2b \
    exp/tri3b
```

## 性能优化技巧

### 内存布局优化

```cpp
// 将所有高斯分量的均值连续存储
struct DiagGmmOptimized {
    Matrix<BaseFloat> means;       // [num_comp x dim]
    Matrix<BaseFloat> vars;        // [num_comp x dim]
    Matrix<BaseFloat> inv_vars;    // [num_comp x dim]
    Vector<BaseFloat> log_dets;    // [num_comp]
    Vector<BaseFloat> weights;     // [num_comp]
};
```

### SIMD 向量化

```cpp
// 使用 AVX2 指令计算多个维度的马氏距离
__m256d ComputeMahalanobisAVX(const __m256d &data,
                               const __m256d &mean,
                               const __m256d &inv_var) {
    __m256d diff = _mm256_sub_pd(data, mean);
    __m256d diff_sq = _mm256_mul_pd(diff, diff);
    return _mm256_mul_pd(diff_sq, inv_var);
}
```

### 预计算

```cpp
// 预计算 log(2π) 和其他常数
const BaseFloat LOG_2PI = log(2.0 * M_PI);

// 预计算每个分量的 log(weight) + (-0.5 * D * log(2π))
Vector<BaseFloat> log_weight_const;
for (int32 i = 0; i < num_comp; i++) {
    log_weight_const(i) = weights_(i) - 0.5 * dim_ * LOG_2PI;
}
```

## GMM 与 DNN 的对比

| 特性 | GMM | DNN |
|------|-----|-----|
| 模型复杂度 | 低 | 高 |
| 特征建模能力 | 线性 | 非线性 |
| 训练数据需求 | 较少 | 大量 |
| 计算效率 | 高 | 中等 |
| 内存占用 | 低 | 高 |
| 当前地位 | 传统方法 | 主流方法 |

## 总结

Kaldi 的 GMM 实现具有以下特点：

1. **高效的对角协方差实现**：通过只存储对角线元素减少内存占用
2. **数值稳定性**：使用对数空间和 log-sum-exp 技巧避免数值下溢
3. **高斯选择优化**：支持多种快速选择策略
4. **完整的训练框架**：包含初始化、EM 算法、模型合并等功能
5. **与 HMM 的无缝集成**：形成 GMM-HMM 混合模型

虽然 DNN 已成为主流，但 GMM 仍然是理解声学建模的基础，也是 Kaldi 训练流程中的重要组成部分。

---

**系列文章目录**：
1. Kaldi 整体架构概览
2. 特征提取模块详解
3. HMM 拓扑与转移模型
4. 高斯混合模型（GMM）实现（本文）
5. 决策树与上下文相关建模
6. 有限状态转换器（FST）集成
7. 解码图构建（HCLG）详解
8. 解码器实现原理
9. 神经网络声学模型
10. 训练流程与工具链
11. Chain 模型与 Lattice-Free MMI
12. 在线解码与流式识别