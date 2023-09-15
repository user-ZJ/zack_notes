# kaldi源码解析6-gmm

gmm模型时比较老的语音模型，需要了解的可以自行百度查看，这里就不展开了，这里主要介绍一下kaldi中gmm各个bin的作用

| gmm的bin                        | 作用                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| gmm-acc-mllt                    | 累积MLLT（全球STC）统计信息<br/>gmm-acc-mllt 1.mdl scp:train.scp ark:1.post 1.macc |
| gmm-acc-mllt-global             | 累积MLLT（全球STC）统计信息：<br/>此版本适用于只有一个全局GMM（例如UBM）的地方<br/>gmm-acc-mllt-global 1.dubm scp:feats.scp 1.macc |
| gmm-acc-stats                   | 累积GMM训练的统计数据（后面step使用）。<br/>gmm-acc-stats 1.mdl scp:train.scp ark:1.post 1.acc |
| gmm-acc-stats2                  | 累积GMM训练的统计数据（后面step使用）。<br/>此版本写两个累加器，将正累加器记为num，将负累加器记为den<br/>gmm-acc-stats2 1.mdl \"$feats\" ark:1.post 1.num_acc 1.den_acc |
| gmm-acc-stats-ali               | 累积GMM训练的统计数据。<br/>gmm-acc-stats-ali 1.mdl scp:train.scp ark:1.ali 1.acc |
| gmm-acc-stats-twofeats          | 累积用于GMM训练的统计信息，使用一组特征计算后验，<br/>而使用另一组特征累积统计信息。 第一项特征用于获得后继者，<br/>第二项用于累积统计数据<br/>gmm-acc-stats-twofeats 1.mdl 1.ali scp:train.scp scp:train_new.scp<br/> ark:1.ali 1.acc |
| gmm-adapt-map                   | 计算一组语音的MAP，如果提供了spk2utt,则计算每个说话人的MAP<br/>通常将结果通过管道传输到gmm-latgen-map中 |
| gmm-align                       | 对齐给定[基于GMM]模型的特征<br/>gmm-align tree 1.mdl lex.fst scp:train.scp <br/>'ark:sym2int.pl -f 2- words.txt text |
| gmm-align-compiled              | 对齐给定[基于GMM]模型的特征<br/>gmm-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.ali |
| gmm-basis-fmllr-accs            | 从训练集中累积每个语音的梯度散射，<br/>如果提供了spk2utt则累积说话人的梯段散射。<br>向后读取以累积每个说话者/话语的fMLLR统计信息，写入梯度散布矩阵。 |
| gmm-basis-fmllr-accs-gpost      | 和gmm-basis-fmllr-accs类似                                   |
| gmm-basis-fmllr-training        | 估计fMLLR基础表示。 读取一组梯度散射累积。 输出基本矩阵      |
| gmm-boost-silence               | 修改基于GMM的模型，以提高（通过某种因素）与指定音素相关联的<br/>所有概率（可以是所有静音音素，或仅用于可选静音的音素）。 <br/>注意：这是通过修改GMM权重来完成的。 如果静默模型与其他<br/>模型共享GMM，则它将修改可能对应于静默的所有模型的GMM权重。 |
| gmm-compute-likes               | 从基于GMM的模型中计算对数似然<br/>（输出由（frame，pdf）索引的对数似然矩阵 |
| gmm-copy                        | 复制基于GMM的模型（并可能更改二进制/文本格式）               |
| gmm-decode-biglm-faster         | 基于GMM的模型解码。 用户提供用于生成解码图的LM和目标LM       |
| gmm-decode-faster               | 基于GMM的模型解码                                            |
| gmm-decode-faster-regtree-fmllr | 基于GMM的模型解码                                            |
| gmm-decode-faster-regtree-mllr  | 基于GMM的模型解码                                            |
| gmm-decode-simple               | 基于GMM的模型解码，维特比解码，仅产生线性序列；<br/> 产生的任何词格都是线性的 |
| gmm-est                         | 对基于GMM的声学模型进行最大似然重估计                        |
| gmm-est-basis-fmllr             | 在测试阶段，针对每个语音执行基本的fMLLR适配。<br/>如果提供了spk2utt则针对每个说话人 <br/>向后读取以累积每个说话者/语音的fMLLR统计信息。 写入矩阵表 |
| gmm-est-basis-fmllr-gpost       | 在测试阶段，针对每个语音执行基本的fMLLR适配。<br/>如果提供了spk2utt则针对每个说话人 <br/>向后读取高斯级后验以累积每个说话者/语音的fMLLR统计信息。 <br/>写入矩阵表 |
| gmm-est-fmllr                   | 在测试阶段，针对每个语音执行基本的fMLLR适配。<br/>如果提供了spk2utt则针对每个说话人 <br/>读取后验（在transition-id上）以累积每个说话者/语音的fMLLR<br/>统计信息。 写入矩阵表 |
| gmm-est-fmllr-global            | 在测试阶段，针对每个语音执行基本的fMLLR适配。<br/>如果提供了spk2utt则针对每个说话人 <br/>此版本适用于只有一个全局GMM的情况，例如 一个UBM。 写入矩阵表 |
| gmm-est-fmllr-gpost             | 在测试阶段，针对每个语音执行基本的fMLLR适配。<br/>如果提供了spk2utt则针对每个说话人 <br/>读高斯水平的后验。 写入矩阵表 |
| gmm-est-fmllr-raw               | 在进行拼接和线性变换（例如LDA + MLLT）之前，先估计空间<br/>中的fMLLR变换，但是在通过这些变换变换的空间中使用模型时，<br/>需要原始的拼接特征以及完整的LDA + MLLT（或类似的）矩阵<br/>（包括“被拒绝”的行） （请参阅程序get-full-lda-mat） |
| gmm-est-fmllr-raw-gpost         | 在进行拼接和线性变换（例如LDA + MLLT）之前，先估计空间<br/>中的fMLLR变换，但是在通过这些变换变换的空间中使用模型时，<br/>需要原始的拼接特征以及完整的LDA + MLLT（或类似的）矩阵<br/>（包括“拒绝”的行） （请参阅程序get-full-lda-mat）。<br/> 阅读高斯水平的后验者。 |
| gmm-est-gaussians-ebw           | 为MMI，MPE或MCE歧视性训练进行EBW更新。 <br/>分子统计信息应已被I平滑处理（例如，使用gmm-ismooth-stats） |
| gmm-est-lvtln-trans             | 估计每个语音或一组说话人的线性VTLN变换（spk2utt选项）        |
| gmm-est-map                     | 对基于GMM的声学模型进行最大后验重估计                        |
| gmm-est-regtree-fmllr           | 计算每个语音/说话人的FMLLR转换                               |
| gmm-est-regtree-fmllr-ali       | 计算每个语音/说话人的FMLLR转换                               |
| gmm-est-regtree-mllr            | 计算每个语音/说话人的MLLR转换                                |
| gmm-est-rescale                 | 对基于GMM的模型进行“重新缩放”重新估计（此更新<br/>会随着特征的变化而改变模型，但会保留模型与特征之间<br/>的差异，以保持任何先前的判别训练的效果）。 在fMPE中使用。<br/> 不更新转换或权重。 |
| gmm-est-weights-ebw             | EBW对MMI，MPE或MCE判别训练的权重进行更新<br/>分子统计数据不执行I-smoothed |
| gmm-fmpe-acc-stats              | 使用GMM模型积累fMPE训练的统计信息                            |
| gmm-get-stats-deriv             | 获取GMM模型的统计衍生数据（用于fMPE / fMMI特征空间判别训练） |
| gmm-global-acc-stats            | 累积用于训练对角协方差GMM的统计信息                          |
| gmm-global-acc-stats-twofeats   | 积累用于训练对角协方差GMM，两特征版本的统计信息<br/>第一项特征用于获得后继者，第二项用于累积统计数据 |
| gmm-global-copy                 | 复制对角协方差GMM                                            |
| gmm-global-est                  | 根据累积的统计量估算对角协方差GMM                            |
| gmm-global-est-fmllr            | 估计全局fMLLR转换。 读取特征和每帧的权重<br/>（使用--weights   --gselect选项） |
| gmm-global-est-lvtln-trans      | 估计每个语音/说话人的线性VTLN变换； 此版本适用于整体<br/>对角线GMM（也称为UBM）。 阅读后验，表明UBM中的高斯指数。 |
| gmm-global-get-frame-likes      | 打印每个语音的每帧对数可能性，作为浮点向量的存档。<br/> 如果--average = true，则将单个语音的平均每帧对数<br/>可能性打印为单个浮点数。 |
| gmm-global-get-post             | 预计算高斯索引并立即转换为前n个后验<br/>（在对角UBM的iVector提取中很有用） |
| gmm-global-gselect-to-post      | 给定特征和对角协方差GMM的高斯选择（gselect）信息，<br/>输出选定索引的每帧后验。 |
| gmm-global-info                 | 将标准GMM模型的各种属性写入标准输出。<br/>这适用于单个对角线GMM，例如 用于UBM。 |
| gmm-global-init-from-feats      | 该程序初始化一个对角线GMM，<br/>并从存储在内存中的特征中进行多次训练。 |
| gmm-global-sum-accs             | 求和多个累积的统计文件，以进行对角协方差GMM训练。            |
| gmm-global-to-fgmm              | 将单个对角协方差GMM转换为单个全协方差GMM。                   |
| gmm-gselect                     | 用于修剪的预计算高斯索引（例如在训练UBM，SGMM，<br/>捆绑混合系统中）对于每个帧，给出n个最佳高斯索引的列表，<br/>从最佳到最差进行排序。 |
| gmm-info                        | 将基于GMM的模型的各种属性写入标准输出                        |
| gmm-init-biphone                | 用所有叶子初始化Biphone上下文相关树（即完整树）。<br/> 适用于端到端无树模型。 |
| gmm-init-lvtln                  | 初始化lvtln转换                                              |
| gmm-init-model                  | 从决策树和树统计信息初始化GMM                                |
| gmm-init-model-flat             | 初始化GMM，并将高斯函数初始化为某些提供的示例数据的<br/>均值和方差（如果未提供，则为0,1：在这种情况下，提供--dim选项） |
| gmm-init-mono                   | 初始化单音GMM。                                              |
| gmm-ismooth-stats               | 将I平滑应用于统计数据，例如 进行区分训练                     |
| gmm-latgen-biglm-faster         | 使用基于GMM的模型生成词格。<br/>用户提供用于生成解码图的LM和目标LM; |
| gmm-latgen-faster               | 使用基于GMM的模型生成词格。                                  |
| gmm-latgen-faster-parallel      | 使用基于GMM的模型生成词格（多线程）                          |
| gmm-latgen-faster-regtree-fmllr | 使用基于GMM的模型和RegTree-FMLLR自适应生成词格。             |
| gmm-latgen-map                  | 使用基于GMM的模型解码功能。                                  |
| gmm-latgen-simple               | 使用基于GMM的模型生成词格。                                  |
| gmm-make-regtree                | 建立回归类树                                                 |
| gmm-mixup                       | GMM是否混合（和高斯合并）                                    |
| gmm-post-to-gpost               | 将state-level后验转换为Gaussian-level后验                    |
| gmm-rescore-lattice             | 使用新模型替换词格上的声学分数。                             |
| gmm-sum-accs                    | 为GMM训练求和多个累积的统计文件。                            |
| gmm-train-lvtln-special         | 将lvtln中的变换之一设置为最小平方误差解，<br/>以将未转换的特征映射到已转换的特征 |
| gmm-transform-means             | 使用线性或仿射变换对GMM进行变换                              |
| gmm-transform-means-global      | 使用线性或仿射变换对GMM进行变换<br/>此版本适用于单个GMM，例如 UBM。在估算MLLT / STC时有用 |



