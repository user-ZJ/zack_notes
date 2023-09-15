# word.3gram.lm和phone.3gram.lm
https://blog.csdn.net/ninesky110/article/details/82179541?utm_source=blogxgwz7

1. 从语料库中生成n-gram计数文件
ngram-count -text train.txt -order 3 -write train.txt.count
-text指向输入文件
-order指向生成几元的n-gram,即n
-write指向输出文件
2. 从上一步生成的计数文件中训练语言模型：
ngram-count -read train.txt.count -order 3 -lm LM -interpolate
-read指向输入文件，为上一步的输出文件
-order与上同
-lm指向训练好的语言模型输出文件
最后两个参数为所采用的平滑方法，-interpolate为插值平滑，-kndiscount为 modified　Kneser-Ney 打折法，这两个是联合使用的，-kndiscount会报错
3. 利用上一步生成的语言模型计算测试集的困惑度：
   ngram -ppl test.txt -order 3 -lm LM >　result
   -ppl为对测试集句子进行评分(logP(T)，其中P(T)为所有句子的概率乘积）和计算测试集困惑度的参数
   result为输出结果文件
   如果想要每条句子单独打分，则使用以下命令：
   ngram -ppl test.txt -order 3 -lm LM -debug 1 >　result

步骤1：统计预料库生成n-gram统计文件
命令行：
ngram-count -vocab segment_dict.txt -text train_data -order 3 -write my.count -unk

segment_dict.txt：词典文件，一行代表一个切词;

train_data:语料库，一行行的数据，行内数据用空格隔开来表示切词
输出文件为
my.count：统计文件

步骤2：生成语言模型
命令行：
ngram-count -read my.count -order 3 -lm train.lm -interpolate

输入为统计文件
输出文件为train.lm：
步骤3：在生成语言模型之后，我们需要将词典文件segment_dict.txt重命名为lexicon，
将语言模型重命名为word.3gram.lm，并将这两个文件防放入thchs30数据集中的lm_word文件
以上为构造lm_word文件内词典和语言模型的过程；

构造lm_phone文件内词典和语言模型的过程同上，只是定义的词典和语料库不同

训练过程中的错误率存放在exp的对应的文件内，比如tri1，tri2b等等

# 术语
LM：language model
KWS：Keyword Search
G2P：Grapheme-to-Phoneme
WSJ：Wall Street Journal

cmvn：倒谱均值和方差归一化
fft：快速傅里叶变换
GMM：高斯混合模型
MFCC：梅尔倒谱系数
pcm：脉冲编码调制
pdf：概率分布函数
PLP：感知线性预测系数
SGMM：子空间高斯混合模型
UBM：通用背景模型
VTLN：特征级声道长度归一化 



# 测试方法：
https://blog.csdn.net/yuansaijie0604/article/details/102601858

 final.mdl是训练出来的模型，words.txt是字典，和HCLG.fst是有限状态机 

 final.mat final.mdl HCLG.fst words.txt 

 final.alimdl final.mat final.mdl full.mat HCLG.fst words.txt 



# thchs30之run.sh粗解
1.数据准备。

最好是那种标注了发音开始和结束时间的语料。但不幸的是，绝大多数开源语料都不满足该条件，必须由算法搞定发音和标签的时间对齐问题。

如果要做声纹识别的话，则语料中还必须有speaker ID。

2.训练monophone模型。

monophone是指那种不包含前后音节的上下文无关的音节模型。它也是后面构建更复杂的上下文相关音节模型的基础。

3.强制对齐（Forced Alignment）。

使用steps/align_si.sh。这里一般使用Viterbi training算法替代传统的Baum-Welch算法。前者的计算效率更高，但精度不如后者。对齐操作可以有效提升标签和语音的吻合度，为后续训练提供便利。

论文：

《Comparative Study of the Baum-Welch and Viterbi Training Algorithms Applied to Read and Spontaneous Speech Recognition》

《Comparative Analysis of Viterbi Training and Maximum Likelihood Estimation for HMMs》

4.用上一步的结果，训练tri1模型（三因素训练）。

这一步很关键，整个thchs30的训练流程，都是用粗糙模型，训练更精细的模型。

5.训练tri2模型（LDA-MLLT特征变换）。

6.训练tri3模型（Speaker Adapted Training，SAT）。主要用到了fMLLR算法。这一步之后的强制对齐使用steps/align_fmllr.sh。

7.训练tri4模型。这一步不再有feature-space的变换，而是对之前特征的综合，比如使用GMM算法。

8.训练DNN模型。

9.被噪声干扰的语音可以使用基于深度自动编码器（DAE）的噪声消除方法。这一步是可选的。tri1 三音素模型是训练与上下文相关的三音子模型；



tri2b 模型用来进行线性判别分析和最大似然线性转换；

tri3b 模型用来训练发音人自适应，基于特征空间最大似然线性回归；

tri4b 模型用来在现有特征上训练模型，它基于树统计中的计数的重叠判断的相似性来选择旧模型中最接近的状态







ark,t:-中的t是IO描述符，IO描述符分为读和写两大类，t是读描述符，表示text。
而-是文件描述符，-表示标准输入输出设备。它也可以是其他命令的输出，例如：
ark:gunzip -c $srcdir/fsts.JOB.gz







# 参考

https://blog.csdn.net/antkillerfarm/article/details/83268422  Kaldi（一）

https://blog.csdn.net/ninesky110/article/details/82179541?utm_source=blogxgwz7 Kaldi中thchs30训练自己数据集的步骤

https://blog.csdn.net/snowdroptulip/article/details/78943748 

https://zhuanlan.zhihu.com/c_1150413643328974848 thchs30脚本详解

https://blog.csdn.net/pelhans/article/details/80003914  加躁训练

