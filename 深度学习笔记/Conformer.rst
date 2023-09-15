Conformer
=======================
Conformer是Google在2020年提出的语音识别模型，基于Transformer改进而来，
主要的改进点在于Transformer在提取长序列依赖的时候更有效，而卷积则擅长提取局部特征，
因此将卷积应用于Transformer的Encoder层，同时提升模型在长期序列和局部特征上的效果

在Conformer论文中，作者指出Conformer相对于Transformer的改进包括：

* 引入深度可分离卷积来提高计算效率。
* 引入更多的正则化方法和dropout技术来提高模型的泛化能力。
* 改进位置编码方法，以便更好地处理长序列。
* 引入音频卷积层，以便更好地处理音频数据。
* 引入新的结构，如卷积层和多头self-attention层的组合。