# kaldi源码解析7-transform

transform中实现了特征变换，有一下几种特征变换：

SAT：说话人自适应

FMLLR：特征空间最大似然线性变换

MLLR:最大似然线性变换(Maximum Likelihood Linear Transform)

CMLLR:Constrained  MLLR,受限的MLLR（等效为fMLLR）

CMVN：倒谱均值归一化

VTLN：声道长度归一化（Vocal Tract Length Normalisation）

lda:降维

fMPE:特征空间区分性训练

Nyquist：为防止信号混叠需要定义最小采样频率，称为奈奎斯特频率

## transform-common

transform-common定义了AffineXformStats用来计算状态转换

```cpp
/// 职业计数器
double beta_;        
/// K_是[均值逆方差]与[扩展数据]的总和，按职业计数进行缩放； 尺寸除以（dim + 1）
Matrix<double> K_;   
///G_是每个维度的扩展数据的外部乘积，按反方差缩放。 这些是fMLLR中的二次统计； 
//在对角fMLLR情况下，G的索引值将为0到dim_-1，但在全fMLLR情况下，
//其索引的索引值将为0到（（dim）（dim + 1））/ 2。 每个G_ [i]的尺寸为dim + 1 x dim + 1。
std::vector< SpMatrix<double> > G_;
/// dim_ is the feature dimension.
int32 dim_;      

```

## cmvn

倒谱均值归一化

```cpp
//创建2x(dim+1)的矩阵，第一行存放均值，最后一个元素为count，第二行存放方差，最后一个元素为0
void InitCmvnStats(int32 dim, Matrix<double> *stats);
//计算单帧的均值和方差（加权）
void AccCmvnStats(const VectorBase<BaseFloat> &feat,BaseFloat weight,
                  MatrixBase<double> *stats);
//计算特征的均值和方差（加权）
void AccCmvnStats(const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> *weights,  // or NULL
                  MatrixBase<double> *stats);
//将倒谱均值和方差归一化应用于特征矩阵。norm_vars表示是否应用方差
void ApplyCmvn(const MatrixBase<double> &stats,bool norm_vars,
               MatrixBase<BaseFloat> *feats);
//倒谱均值和方差归一化的方向操作，即乘方差后再加均值
void ApplyCmvnReverse(const MatrixBase<double> &stats,bool norm_vars,
                      MatrixBase<BaseFloat> *feats);
//修改统计信息,以便对某些维度,将其替换为均值和单位方差为零的“伪”统计信息,这样做是为了禁用这些维度的CMVN。
void FakeStatsForSomeDims(const std::vector<int32> &dims,MatrixBase<double> *stats);
```

## basis-fmllr-diag-gmm

gmm特征空间最大似然线性变换的基础方法

BasisFmllrAccus：fMLLR子空间估计的统计信息。训练阶段使用

```cpp
//计算梯度
void AccuGradientScatter(const AffineXformStats &spk_stats);
```

BasisFmllrEstimate：基本fMLLR的估算，测试阶段使用

```cpp
//以最大似然方式有效地估计基本矩阵
void EstimateFmllrBasis(const AmDiagGmm &am_gmm,const BasisFmllrAccus &basis_accus);
//在基础矩阵估计之前计算预处理器矩阵
void ComputeAmDiagPrecond(const AmDiagGmm &am_gmm,SpMatrix<double> *pre_cond);
//说话人自适应，根据说话人统计信息计算fMLLR矩阵。
double ComputeTransform(const AffineXformStats &spk_stats,Matrix<BaseFloat> *out_xform,
                       Vector<BaseFloat> *coefficients,BasisFmllrOptions options) const;
```

## compressed-transform-stats

将AffineXformStats压缩到更少的内存中，以便更轻松地在网络上进行存储和传输.

## decodable-am-diag-gmm-regtree

gmm解码用，暂不介绍

## fmllr-diag-gmm

从gmm进行特征空间最大似然线性变换

## fmllr-raw

在mfcc或类似特征上进行拼接和投影时计算fmllr.

## fmpe

特征空间区分性训练

## lda-estimate

LdaEstimate：用于计算线性判别分析（LDA）变换的类。

```cpp
void Init(int32 num_classes, int32 dimension); //为累加器分配内存
void ZeroAccumulators();  //将所有累加器设置为零
void Scale(BaseFloat f);  //缩放所有累加器
double TotCount();  //返回数据的总数
//根据数据累积
void Accumulate(const VectorBase<BaseFloat> &data, int32 class_id, BaseFloat weight=1.0);
//估计LDA变换矩阵m
void Estimate(const LdaEstimateOptions &opts, 
                Matrix<BaseFloat> *M,Matrix<BaseFloat> *Mfull = NULL) const;
```

## lvtln

将线性近似应用于VTLN变换。

## mllt

估计GMM的最大似然线性变换，也称为全局半联合协方差（STC），所得的变换将特征向量左乘。

## regression-tree

回归树是声学模型中高斯密度的聚类，因此树的每个节点处的高斯组通过相同的变换进行变换。 每个节点因此称为回归类。







