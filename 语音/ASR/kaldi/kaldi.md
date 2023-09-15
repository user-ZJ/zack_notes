# kaldi



```cpp
//数据读入类型
enum RspecifierType  {
  kNoRspecifier,  错误
  kArchiveRspecifier, ark
  kScriptRspecifier  scp  
};
```

```
指令前缀
o (once), no (not-once)，s (sorted), ns (not-sorted), p (permissive),np (not-permissive),b(binary),t(text)
ark:rxfilename  ->  kArchiveRspecifier
scp:rxfilename  -> kScriptRspecifier
b, ark:rxfilename  ->  kArchiveRspecifier
t, ark:rxfilename  ->  kArchiveRspecifier
b, scp:rxfilename  -> kScriptRspecifier
t, no, s, scp:rxfilename  -> kScriptRspecifier
t, ns, scp:rxfilename  -> kScriptRspecifier
```

```cpp
scp文件读取状态
enum StateType {
    kUninitialized, // no  no  no  no     未初始化或关闭的对象.
    kFileStart,     // no  no  yes no     只是打开了scp文件
    kEof,           // no  no  no  no      调用Next() 到达scp文件结尾.
    kError,         // no  no  no  no      读scp文件错误.
    kHaveScpLine,   // no  no  yes yes   一行scp文件，但没有其他内容.
    kHaveObject,    // yes no  yes yes   holder_ 包含对象但是 range_holder_ 不包含.
    kHaveRange,     // yes yes yes yes   在range_holder_中包含range对象 
  } state_;
// no  no  no  no 四个标志分别表示
//holder_是否包含data_rxfilename_ 对象
//只是打开.scp
//range_holder_ 是否包含range对象
//script_input_ （scp）是否打开
//设置了key_, data_rxfilename_ 和 range_ [如果适用]
```

```cpp
ark文件读取状态
enum StateType { 
    kUninitialized,  // 未初始化或关闭.                  no         no
    kFileStart,      // 刚刚打开                        no         yes
    kEof,     // 调用next()到达文件末尾                   no         no
    kError,   // Some other error                       no         no
    kHaveObject,  // 读出 key 和 object                  yes        yes
    kFreedObject,  // 用户调用FreeCurrent().              no         yes
  } state_;
//两个状态分别表示
//holder_是否有对象
//input_是否打开
```

SequentialTableReaderImplBase  顺序表，基础类

SequentialTableReaderScriptImpl  读scp

SequentialTableReaderArchiveImpl 读ark

SequentialTableReaderBackgroundImpl 后台执行

SequentialTableReader    适配器类，根据传入数据不同，选择不同的reader

TableWriterImplBase  基础类

TableWriterArchiveImpl  直接写入没有关联scp的ark时使用的TableWriter的实现

TableWriterScriptImpl    写scp

TableWriterBothImpl      写scp和ark文件

RandomAccessTableReaderImplBase   随机读，基础类

RandomAccessTableReaderScriptImpl  读scp，继承RandomAccessTableReaderImplBase   

RandomAccessTableReaderArchiveImplBase  读ark，继承RandomAccessTableReaderImplBase   

RandomAccessTableReaderDSortedArchiveImpl  called_sorted(cs) DSorted for "doubly sorted"，继承RandomAccessTableReaderArchiveImplBase  

RandomAccessTableReaderSortedArchiveImpl  sorted (s)，继承RandomAccessTableReaderArchiveImplBase  

RandomAccessTableReaderUnsortedArchiveImpl    继承RandomAccessTableReaderArchiveImplBase  

RandomAccessTableReaderMapped  utt2spk+scp



```cpp
写入数据类型
enum WspecifierType  {
  kNoWspecifier,错误
  kArchiveWspecifier, ark 
  kScriptWspecifier,  scp
  kBothWspecifier  ark,scp
};

```

```
ark,t:wxfilename -> kArchiveWspecifier
ark,b:wxfilename -> kArchiveWspecifier
scp,t:rxfilename -> kScriptWspecifier
scp,t:rxfilename -> kScriptWspecifier
ark,scp,t:filename, wxfilename -> kBothWspecifier
ark,scp:filename, wxfilename ->  kBothWspecifier
f(flush),nf(no-flush),b(binary),t(text)
不允许"scp, ark", 只允许 "ark,scp"
```

### 表单属性

1. 写表单属性
   * 表单类型：scp；ark；ark,scp
   * 二进制模式：b
   * 文本模式：t
   * 刷新模式：f；nf。用于确定在每次写操作后是否刷新数据流，默认是f，即刷新，这样有利于优化内存使用
   * 宽容模式：p。只对输出列表表单(scp)有效。例如，在同时输出存档表单和列表表单时，如果表单的某个元素对应的存档内容无法获取，那么在列表表单中将直接跳过这个元素，不提示错误。
2. 读表单属性
   * 表单类型：scp；ark。
   * 单次访问：o；no。告知可执行程序，在读入表单中每个索引只出现一次，不会出现多个元素使用一个索引的情况
   * 宽容模式：p；np。如果输入的列表表单中某个元素的目标文件无法获取或输入的存档表单中某个元素的内容有误，则不会抛出错误，而是在日志中打印一个警告
   * 有序表单：s；ns。在输入的表单中，元素的索引是有序的，这个有序是字符串意义上的
   * 有序访问：cs；ncs。表单中的元素将被顺序访问
   * 存储格式：b；t。在最新设计中列表表单只能是文本，存档表单可能由'\0B'来判断，所以这个读属性已经不在使用。
   * 后台执行：bg,在后台线程中“提前读取”下一个值。在读取较大的对象（例如神经网络训练示例）时，尤其在要最大程度地利用GPU的情况下，建议使用此选项。

## 对齐
对齐(align)或强制对齐(forced alignment)目的是获取每一帧所对应的状态。
做法是根据文本生成状态图，使用生成的状态图进行ASR识别（把解码路径限制在生成的直线型状态图中，使得识别结果必定是参考文本）


## HCLG
HCLG = H o C o L o G

* G （grammar）是用来编码语法或者语言模型的接收器(i.e：输入是拼音，输出是句子)
* L 是发声词典（lexicon）；输出是拼音，输入是音素。
* C 表示上下文关系（context），输出是音素，输入是上下文相关的音素例如N个音
  素组成的窗。具体看Phonetic context windows.里面的介绍。
*  H 包含HMM 的定义；输出符表示上下文相关的音素，输入是
  transitions-ids(转移id)，transitions-ids 是编码pdf-id 或者其他信息(自转或
  向后转)，具体看( Integer identifiers used by TransitionModel)

## FAQ

### 1. 消除mfcc fbank plp特征提取中的随机性,且不影响精度

解决方案：在compute-mfcc-feats.cc的主函数中加入srand(0)