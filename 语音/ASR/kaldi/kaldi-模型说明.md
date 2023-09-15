# 模型说明

## GMM-HMM[声学模型](https://zh.wikipedia.org/wiki/%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B)
模型由一个**TransitionModel**和多个**DiagGMM**构成。   
TransitionModel是声学模型的头部，存储Transition模型，定义每个音素由多少个状态构成等信息。   
DiagGMM描述状态概率分布，每个DiagGMM为一个状态的高斯分量的概率分布函数(也经常被称为一个pdf)，内容由MEANS_INVVARS,INV_VARS,WEIGHTS,GCONSTS四部分构成。为了减少实时计算量，kaldi并不是直接存储这些参数，而是用这些参数做了一些概率密度的预计算（如矩阵求逆等），把预计算结果存储在模型中。    

### TransitionModel
TransitionModel分为Topology,Triples,LogProbs三部分。   
**Topology**：表示各音素由多少个状态构成，每个状态之间的转换概率是多少；
**Triples**：由众多三元组构成，每个三元组的定义为(音素索引，HMM状态，PDF索引)。把全部的这些三元组放在一起，从1开始编号，每个编号对应一个transition state。   
transition state有若干可能的跳转指向其他状态，对这些跳转从0开始编号，这样就得到transition-index。
把(transition-state,transition-index)作为一个二元组并从1开始编号，该编号就被称为transition-id。   
**LogProbs**：对数转移概率(以e为底)向量，这个向量按transition-id索引，由于transition-id从1开始，所以LogProbs向量在前面补0

**transition-id能够唯一地映射成音素和pdf-id,所以kaldi中使用transition-id作为对齐结果，并作为HCLG的输入标签**
**每个PDF索引对应一组高斯混合模型参数**


HMM state表示不同音素状态
HMM模型本质上是一个序列分类器sequence classifier，就是把一个某长度的序列识别成另一个长度的序列，比如把一系列MFCC特征正确的识别成对应HMM state序列
这个过程涉及两个概率需要学习，一是把当前frame的特征识别为这个state的概率(也就是GMM中的mean vector 和covariance matrix )，二是上个state转化为这个state的概率也就是状态转移概率Transition probabilities。
一个序列转化为另一个序列理论上有指数级种转化方式，所以每一个frame只取概率最高的那个state，这样的路线选择方法被称为Viterbi 方法。HMM的训练过程就是利用每个训练样本以及其对应的句子采用Viterbi 方法不断迭代更新GMM中每个state的mean vector和covariance matrix以及它们的状态转移概率，最后达到收敛。



gmm-hmm就是把MFCC特征用混合高斯模型区模拟，然后把均值和方差输入到hmm的模型里。

马尔可夫模型的概念是一个离散时域有限状态自动机，隐马尔可夫模型HMM是指这一马尔可夫模型的内部状态外界不可见，外界只能看到各个时刻的输出值。
对语音识别系统，输出值通常就是从各个帧计算而得的声学特征。
用HMM刻画语音信号需作出两个假设，一是内部状态的转移只与上一状态有关，另一是输出值只与当前状态（或当前的状态转移）有关，这两个假设大大降低了模型的复杂度。


声学模型的输入是由特征提取模块提取的特征(MFCC)。
