# readme

重点：https://github.com/espnet/espnet  

https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch  

https://github.com/HawkAaron/E2E-ASR   

https://github.com/facebookresearch/wav2letter   

https://github.com/freewym/espresso   

https://github.com/pytorch/fairseq/tree/master/examples/speech_recognition  

https://github.com/jitsi/asr-wer   

https://github.com/syhw/wer_are_we   

https://github.com/mdangschat/ctc-asr   

https://github.com/ekapolc/ASR_course   

https://github.com/gooofy/zamia-speech   





https://www.jiqizhixin.com/articles/021102

重点：https://github.com/nl8590687/ASRT_SpeechRecognition

https://github.com/fighting41love/funNLP



https://www.jishuwen.com/d/2uyS 

DNN Hybird Acoustic Models

Recurrent DNN Hybird Acoustic Models

CNN-CTC

GRU-CTC

RNN-CTC

CNN-RNN-CTC

CLDNN

Deep CNN

FSMN

DFSMN

DFCNN 科大讯飞提出的称为全序列卷积神经网络（deep fully convolutional
neural network，DFCNN）模型

deep speech



https://zhuanlan.zhihu.com/p/48729548

现在主流的利用深度学习的语音识别模型中仍在存在多种派系:

一种是利用深度学习模型取代原来的GMM部分，即DNN-HMM类的模型，需要先实现HMM结构与语音的对齐然后才能进一步地训练深度神经网络，除此之外，在训练这一类的模型时，训练样本的标注不仅仅是原本的文本，还需要对文本进一步的拆解成为音素投入训练，这对于标注部分的工作就会造成极大的挑战。在解码的时候，这种模型同样还需要依赖这个发音词典。

另一种则是作者采用的端到端的深度学习模型，端到端的模型旨在一步直接实现语音的输入与解码识别，从而不需要繁杂的对齐工作与发音词典制作工作，具有了可以节省大量的前期准备时间的优势，真正的做到数据拿来就可用。端到端的模型的另一个优点是，更换识别语言体系时可以利用相同的框架结构直接训练。例如同样的网络结构可以训练包含26个字符的英文模型，也可以训练包含3000个常用汉字的中文模型，甚至可以将中英文的词典直接合在一起，训练一个混合模型。





纯CTC解码通过预测每个帧的输出来识别语音，算法的实现基于假设每帧的解码保持彼此独立，因而缺乏解码过程中前后语音特征之间的联系，比较依赖语言模型的修正

纯attention解码过程则与输入语音的帧的顺序无关，每个解码单元是通过前一单元的解码结果与整体语音特征来生成当前的结果，解码过程忽略了语音的单调时序性。



https://blog.csdn.net/chinatelecom08/article/details/82557715





kaldi中文ASR：

https://taylorguo.gitbooks.io/asr-guide/kaldiru-men.html  

nvidia-smi -c 3



## CMUSphinx安装
https://blog.csdn.net/xj853663557/article/details/84583973



## ngram使用

```shell
#kaldi dummies doc
ngram-count -order 1 -write-vocab $local/tmp/vacab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa
arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst

#lm生成
ngram-count -text word_cropus.txt -order 1 -write train.txt.count -unk
ngram-count -read train.txt.count -order 1 lm word.1gram.lm -interpolate


```



jsgf转G.fst

There is a sphinx tool called "sphinx_jsgf2fsg" which can be used to convert jsfg to fsm. You can then use fst tools to create a G.fst 

sphinx_jsgf2fsg -jsgf test.jsgf -fsm test.fsm  



use "fstcompile" to convert it from text to the binary G.fst format that can be incorporated into HCLG.fst and used for decoding



https://github.com/alumae/kaldi-gstreamer-server/issues/122

sphinx_jsgf2fsg -jsgf G.jsgf -fsm G.fsm                            
fstcompile --acceptor --isymbols=$lang/words.txt --osymbols=$lang/words.txt --keep_isymbols=false --keep_osymbols=false G.fsm \| fstrmepsilon > $lang/G.fst 



```
#JSGF V1.0;

grammar digit;
<number> = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;
public <fourdigit> = <number> | <number> <number> | <number> <number> <number> | <number> <number> <number> <number>;
```





输出语音时间：

https://groups.google.com/forum/#!topic/kaldi-help/Fye4gkL51T0

steps/align_si.sh --nj 1 --cmd [run.pl](http://run.pl/) data/test data/lang exp/mono exp/test/mono_ali

steps/get_train_ctm_mod.sh --cmd [run.pl](http://run.pl/) --stage 0 --use-segments false ./data/test ./data/lang ./exp/test/mono_ali

exp/test/mono_ali/ctm



lattice-1best ark:DECODE_FOLDER/lat.1 ark:- | lattice-align-words-lexicon data/lang/phones/align_lexicon.int MODEL_PATH/final.mdl ark:- ark:- | nbest-to-ctm ark:- - | utils/int2sym.pl -f 5LANG_PATH/words.txt > 1.ctm





pitch：指的是基频

plp:感知线性预测





#Snt是句子个数，#Wrd是单词个数。Corr是词正确率，Sub是替换错误率，Del是删除错误率, Ins是插入错误率, Err是整体错误率，S.Err总体的句子错误率







kaldi GPU 解码：

https://developer.nvidia.com/blog/gpu-accelerated-speech-to-text-with-kaldi-a-tutorial-on-getting-started/